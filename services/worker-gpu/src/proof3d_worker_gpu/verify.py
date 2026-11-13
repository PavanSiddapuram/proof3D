"""Verify (PRD 5.4), the silhouette part that runs on CPU today.

For every model of the job (an imported one, or later A to D) it runs ``check_model``: a camera
fit per usable photo, silhouette IoU, surface evidence and next-photo suggestions (ADR 0007).
It writes one score per held-out photo, the evidence on the asset version, and the full report
to storage.

Only silhouettes are scored. Rendering the textured model and scoring appearance (LPIPS,
DINOv2, FLIP, edges) needs the GPU path, so a score never passes yet: ``passed`` is False when
a held-out silhouette disagrees and None otherwise, never True on silhouette alone.
"""

from __future__ import annotations

import io
import json
from dataclasses import asdict
from pathlib import PurePosixPath

import numpy as np
import trimesh
from sqlalchemy import delete, select

from proof3d_api.db import AssetVersion, Photo, Product, Score
from proof3d_api.worker import StageContext, StageError
from proof3d_geometry.camera import FloatArray
from proof3d_geometry.camera_fit import FitSettings
from proof3d_geometry.check import ModelCheck, PhotoInput, check_model
from proof3d_geometry.raster import IntArray
from proof3d_pipeline import PhotoRole, StageName, StageResult, StageStatus
from proof3d_pipeline.config import CameraFitConfig
from proof3d_pipeline.measure import measure
from proof3d_pipeline.storage import Storage, content_key
from proof3d_worker_gpu.ingest import decode_mask, photo_name

SCORED = ["silhouette_iou"]
NOT_SCORED_YET = ["lpips", "dino", "flip", "edges"]
_USABLE = {PhotoRole.INPUT.value, PhotoRole.HELD_OUT.value}


def fit_settings(cfg: CameraFitConfig) -> FitSettings:
    return FitSettings(**cfg.model_dump(exclude={"max_fit_faces"}))


def load_mesh(storage: Storage, key: str) -> tuple[FloatArray, IntArray]:
    if "://" in key:
        raise StageError(
            "The model is a URL, and fetching models from URLs is not built yet. Upload the GLB "
            "with POST /uploads and import the returned key."
        )
    if not storage.exists(key):
        raise StageError("The model file is not in storage; upload it again.")
    file_type = PurePosixPath(key).suffix.lstrip(".").lower() or "glb"
    try:
        mesh = trimesh.load(io.BytesIO(storage.get(key)), file_type=file_type, force="mesh")
    except Exception as exc:  # trimesh raises many types for a bad file
        raise StageError(f"The model file could not be read as {file_type.upper()}.") from exc
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise StageError("The model file has no triangles.")
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)


def _photo_input(storage: Storage, photo: Photo) -> PhotoInput:
    assert photo.mask_key is not None, "only photos with masks are verified"
    return PhotoInput(
        photo_name(photo),
        decode_mask(storage.get(photo.mask_key)),
        photo.view_label,
        held_out=photo.role == PhotoRole.HELD_OUT.value,
    )


def verdict(report: ModelCheck) -> str:
    if report.held_out_disagreements:
        return "disagrees"
    if not any(p.held_out and not p.low_iou for p in report.photos):
        return "not_checkable"
    return "silhouette_agrees"


def _record(
    ctx: StageContext, asset: AssetVersion, photos: list[Photo], report: ModelCheck, faces: int
) -> str:
    body = json.dumps(report.to_dict(), indent=2, sort_keys=True).encode()
    report_key = ctx.storage.put(content_key(f"reports/{asset.id}/verify", body, ".json"), body)

    # A rerun replaces this asset's scores rather than adding to them.
    ctx.session.execute(delete(Score).where(Score.asset_id == asset.id))
    input_fits = []
    for photo, check in zip(photos, report.photos, strict=True):
        if not check.held_out:
            input_fits.append(
                {
                    "photo_id": photo.id,
                    "name": check.name,
                    "fit_iou": check.silhouette_iou,
                    "low_iou": check.low_iou,
                    "ambiguous": check.ambiguous,
                    "surface_shown": check.surface_shown,
                }
            )
            continue
        ctx.session.add(
            Score(
                asset_id=asset.id,
                photo_id=photo.id,
                fit_iou=check.silhouette_iou,
                fit_ambiguous=check.ambiguous,
                metrics={
                    "silhouette_iou": check.silhouette_iou,
                    "surface_shown": check.surface_shown,
                },
                passed=False if check.low_iou else None,
                render_keys={"report": report_key},
            )
        )
    asset.unverified_surface_share = 1.0 - report.checkable_share
    asset.triangles = asset.triangles or faces
    asset.evidence = {
        "checkable_share": report.checkable_share,
        "evidenced_share": report.evidenced_share,
        "invented_share": report.invented_share,
        "faces_checked": report.faces_used,
        "next_photos": [asdict(s) for s in report.suggestions],
        "input_fits": input_fits,
        "notes": report.notes,
        "report_key": report_key,
    }
    ctx.session.flush()
    ctx.session.expire(asset, ["scores"])
    return report_key


def verify(ctx: StageContext) -> StageResult:
    with measure() as meter:  # CPU work, even when the GPU worker runs it
        assets = ctx.session.scalars(
            select(AssetVersion)
            .where(AssetVersion.job_id == ctx.job_id)
            .order_by(AssetVersion.created_at)
        ).all()
        if not assets:
            raise StageError("Nothing to verify: no model was generated or imported for this job.")
        product = ctx.session.get(Product, ctx.product_id)
        assert product is not None, "a job always has a product"
        photos = sorted(
            (p for p in product.photos if p.role in _USABLE and p.mask_key),
            key=lambda p: (photo_name(p), p.source),
        )
        if not photos:
            raise StageError("No usable photos with masks. Rerun from ingest.")
        inputs = [_photo_input(ctx.storage, p) for p in photos]
        fit_cfg = ctx.config.verify.camera_fit

        outputs: dict[str, str] = {}
        results = []
        for asset in assets:
            vertices, faces = load_mesh(ctx.storage, asset.glb_key)
            report = check_model(
                vertices, faces, inputs, fit_settings(fit_cfg), max_faces=fit_cfg.max_fit_faces
            )
            outputs[f"report/{asset.id}"] = _record(ctx, asset, photos, report, len(faces))
            results.append(
                {
                    "asset_id": asset.id,
                    "method": asset.method,
                    "verdict": verdict(report),
                    "disagreements": report.held_out_disagreements,
                    "checkable_share": report.checkable_share,
                    "invented_share": report.invented_share,
                }
            )

    return StageResult(
        stage=StageName.VERIFY,
        status=StageStatus.SUCCEEDED,
        metrics=meter.metrics(),
        outputs=outputs,
        summary={"results": results, "scored": SCORED, "not_scored_yet": NOT_SCORED_YET},
    )
