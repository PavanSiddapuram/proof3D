"""HTTP API (PRD 4.3).

No `from __future__ import annotations` here: FastAPI must resolve the locally defined `DB`
dependency alias at definition time.
"""

import hashlib
import re
from collections.abc import Iterator
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from proof3d_api import queue
from proof3d_api.db import AssetVersion, Base, Job, Photo, Product, make_engine, make_sessionmaker
from proof3d_api.schemas import (
    CompareOut,
    EvidenceOut,
    ImportRequest,
    ImportResponse,
    JobOut,
    ScorecardOut,
    ScoreOut,
    StageOut,
    UploadOut,
    Verdict,
)
from proof3d_api.settings import Settings, get_settings, make_storage
from proof3d_pipeline import (
    Condition,
    PhotoRole,
    PipelineConfig,
    StageName,
    StageStatus,
    config_hash,
)
from proof3d_pipeline.storage import Storage

# Accepted uploads, recognised by their first bytes rather than the name or Content-Type.
_SIGNATURES: tuple[tuple[bytes, Literal["image", "model"], str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image", ".png"),
    (b"\xff\xd8\xff", "image", ".jpg"),
    (b"glTF", "model", ".glb"),
)


def create_app(
    settings: Settings | None = None,
    create_tables: bool | None = None,
    storage: Storage | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    store = storage or make_storage(settings)
    engine = make_engine(settings.database_url)
    if settings.create_tables if create_tables is None else create_tables:
        Base.metadata.create_all(engine)
    sessions: sessionmaker[Any] = make_sessionmaker(engine)

    app = FastAPI(title="Proof3D", version=settings.pipeline_version)

    def db() -> Iterator[Session]:
        with sessions() as session, session.begin():
            yield session

    DB = Annotated[Session, Depends(db)]

    @app.post("/uploads", response_model=UploadOut, status_code=201)
    async def upload(
        request: Request, filename: Annotated[str, Query(min_length=1, max_length=128)]
    ) -> UploadOut:
        """Store a photo (PNG, JPEG) or model (GLB) sent as the raw request body. The key is
        content-addressed, so uploading the same file twice gives the same key."""
        limit = settings.max_upload_mb * 2**20
        too_large = HTTPException(413, f"uploads are limited to {settings.max_upload_mb} MB")
        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > limit:
            raise too_large
        chunks: list[bytes] = []
        size = 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > limit:
                raise too_large
            chunks.append(chunk)
        data = b"".join(chunks)
        match = next((s for s in _SIGNATURES if data.startswith(s[0])), None)
        if match is None:
            raise HTTPException(415, "only PNG, JPEG and GLB files are accepted")
        _, kind, suffix = match
        key = f"uploads/{hashlib.sha256(data).hexdigest()}/{_safe_name(filename, suffix)}"
        await run_in_threadpool(store.put, key, data)
        return UploadOut(key=key, bytes=size, kind=kind)

    @app.post("/products/import", response_model=ImportResponse, status_code=201)
    def import_product(req: ImportRequest, session: DB) -> ImportResponse:
        try:
            cfg = PipelineConfig.model_validate(req.config)
        except ValueError as exc:
            raise HTTPException(422, f"invalid pipeline config: {exc}") from exc
        # URLs are fetched by ingest; storage keys must already be uploaded.
        files = [*req.photos, *([req.model.glb] if req.model else [])]
        missing = [k for k in files if "://" not in k and not _stored(store, k)]
        if missing:
            raise HTTPException(422, f"not in storage (upload with POST /uploads): {missing}")
        product = Product(
            source_url=str(req.source_url) if req.source_url else None,
            title=req.title,
            category=req.category,
        )
        external = req.model
        held_out = set(external.held_out) if external else set()
        product.photos = [
            Photo(
                source=src,
                # Only the importer knows which photos the generator saw, so roles are set
                # here; ingest keeps them and only adds masks, labels and quality flags.
                role=(PhotoRole.HELD_OUT if src in held_out else PhotoRole.INPUT).value
                if external
                else None,
            )
            for src in req.photos
        ]
        job = Job(
            product=product,
            pipeline_version=settings.pipeline_version,
            config=cfg.model_dump(mode="json"),
            config_hash=config_hash(cfg),
        )
        # An imported model is verified as it is: no generation, retexturing or repair.
        skip = (
            {StageName.GENERATE_SHAPE, StageName.PROJECT_TEXTURE, StageName.REPAIR}
            if external
            else set()
        )
        queue.create_stage_runs(job, skip=skip)
        session.add_all([product, job])
        session.flush()
        asset_id = None
        if external:
            asset = AssetVersion(
                job_id=job.id,
                method=Condition.EXTERNAL.value,
                glb_key=external.glb,
                provenance={
                    "generator": external.generator,
                    "generator_version": external.generator_version,
                    "imported": True,
                },
            )
            session.add(asset)
            session.flush()
            asset_id = asset.id
        return ImportResponse(job_id=job.id, product_id=product.id, asset_id=asset_id)

    @app.get("/jobs/{job_id}", response_model=JobOut)
    def get_job(job_id: str, session: DB) -> JobOut:
        job = session.get(Job, job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        return _job_out(job)

    @app.post("/jobs/{job_id}/rerun", response_model=JobOut, status_code=202)
    def rerun(job_id: str, session: DB, from_: Annotated[StageName, Query(alias="from")]) -> JobOut:
        job = session.get(Job, job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        try:
            queue.rerun_from(session, job, from_)
        except queue.RerunConflictError as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
        return _job_out(job)

    @app.get("/assets/{asset_id}/scorecard", response_model=ScorecardOut)
    def scorecard(asset_id: str, session: DB) -> ScorecardOut:
        asset = session.get(AssetVersion, asset_id)
        if asset is None:
            raise HTTPException(404, "asset not found")
        return _scorecard(asset)

    @app.get("/products/{product_id}/compare", response_model=CompareOut)
    def compare(product_id: str, session: DB) -> CompareOut:
        if session.get(Product, product_id) is None:
            raise HTTPException(404, "product not found")
        assets = session.scalars(
            select(AssetVersion)
            .join(Job)
            .where(Job.product_id == product_id)
            .order_by(AssetVersion.created_at)
        ).all()
        latest: dict[str, AssetVersion] = {}
        for a in assets:  # later versions of a condition replace earlier ones
            latest[a.method] = a
        return CompareOut(
            product_id=product_id, conditions={m: _scorecard(a) for m, a in sorted(latest.items())}
        )

    @app.post("/assets/{asset_id}/publish", response_model=JobOut, status_code=202)
    def publish(asset_id: str, session: DB) -> JobOut:
        asset = session.get(AssetVersion, asset_id)
        if asset is None:
            raise HTTPException(404, "asset not found")
        job = session.get(Job, asset.job_id)
        assert job is not None
        try:
            queue.rerun_from(session, job, StageName.PUBLISH)
        except queue.RerunConflictError as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
        return _job_out(job)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app


def _job_status(job: Job) -> str:
    statuses = [s.status for s in job.stages]
    if StageStatus.FAILED.value in statuses:
        return StageStatus.FAILED.value
    if StageStatus.RUNNING.value in statuses:
        return StageStatus.RUNNING.value
    if all(s in (StageStatus.SUCCEEDED.value, StageStatus.SKIPPED.value) for s in statuses):
        return StageStatus.SUCCEEDED.value
    return StageStatus.QUEUED.value


def _job_out(job: Job) -> JobOut:
    costs = [s.cost_usd for s in job.stages if s.cost_usd is not None]
    return JobOut(
        id=job.id,
        product_id=job.product_id,
        pipeline_version=job.pipeline_version,
        config_hash=job.config_hash,
        status=_job_status(job),
        stages=[StageOut.model_validate(s) for s in job.stages],
        total_wall_s=sum(s.wall_s or 0.0 for s in job.stages),
        total_gpu_s=sum(s.gpu_s or 0.0 for s in job.stages),
        # Unknown cost stays unknown rather than being reported as zero.
        total_cost_usd=sum(costs) if costs else None,
    )


def _safe_name(filename: str, suffix: str) -> str:
    """A storage-safe file name carrying the extension of the detected type."""
    stem = PurePosixPath(filename.replace("\\", "/")).stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")[:80]
    return f"{stem or 'upload'}{suffix}"


def _stored(store: Storage, key: str) -> bool:
    try:
        return store.exists(key)
    except ValueError:  # malformed, or escapes the storage root
        return False


def _verdict(asset: AssetVersion) -> Verdict | None:
    if asset.evidence is None:
        return None
    if not asset.scores:
        return "not_checkable"
    if any(s.passed is False for s in asset.scores):
        return "disagrees"
    return "silhouette_agrees"


def _scorecard(asset: AssetVersion) -> ScorecardOut:
    return ScorecardOut(
        asset_id=asset.id,
        method=asset.method,
        verdict=_verdict(asset),
        unverified_surface_share=asset.unverified_surface_share,
        evidence=EvidenceOut.model_validate(asset.evidence) if asset.evidence else None,
        triangles=asset.triangles,
        bytes=asset.bytes,
        scores=[
            ScoreOut.model_validate(s).model_copy(
                update={"photo_name": PurePosixPath(s.photo.source).name}
            )
            for s in asset.scores
        ],
        provenance=asset.provenance,
    )
