"""Relational model for PRD 4.2. Jobs, scores, versions and device runs are all joined on, so
they live in Postgres; binary outputs live in object storage and are referenced by key."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, ClassVar

from sqlalchemy import JSON, ForeignKey, String, UniqueConstraint, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from proof3d_pipeline import StageStatus


def _id() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    type_annotation_map: ClassVar[dict[Any, Any]] = {dict[str, Any]: JSON, list[Any]: JSON}


class Product(Base):
    __tablename__ = "products"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_id)
    source_url: Mapped[str | None]
    category: Mapped[str | None]
    title: Mapped[str | None]
    created_at: Mapped[datetime] = mapped_column(default=_now)
    photos: Mapped[list[Photo]] = relationship(back_populates="product")
    jobs: Mapped[list[Job]] = relationship(back_populates="product")


class Photo(Base):
    __tablename__ = "photos"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_id)
    product_id: Mapped[str] = mapped_column(ForeignKey("products.id"))
    source: Mapped[str]  # URL or storage key of the original image
    role: Mapped[str | None]  # PhotoRole, set by ingest
    mask_key: Mapped[str | None]
    view_label: Mapped[str | None]
    camera: Mapped[dict[str, Any] | None]  # estimated intrinsics/extrinsics
    quality_flags: Mapped[list[Any]] = mapped_column(default=list)
    product: Mapped[Product] = relationship(back_populates="photos")


class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_id)
    product_id: Mapped[str] = mapped_column(ForeignKey("products.id"))
    pipeline_version: Mapped[str]
    config: Mapped[dict[str, Any]]
    config_hash: Mapped[str]
    created_at: Mapped[datetime] = mapped_column(default=_now)
    product: Mapped[Product] = relationship(back_populates="jobs")
    stages: Mapped[list[StageRun]] = relationship(
        back_populates="job", order_by="StageRun.position", cascade="all, delete-orphan"
    )


class StageRun(Base):
    """One stage of one job. Doubles as the queue: workers claim QUEUED rows with
    ``SELECT ... FOR UPDATE SKIP LOCKED`` (ADR 0002)."""

    __tablename__ = "stage_runs"
    __table_args__ = (UniqueConstraint("job_id", "stage"),)
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_id)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), index=True)
    stage: Mapped[str]
    position: Mapped[int]
    worker: Mapped[str] = mapped_column(index=True)
    status: Mapped[str] = mapped_column(default=StageStatus.PENDING.value, index=True)
    attempts: Mapped[int] = mapped_column(default=0)
    queued_at: Mapped[datetime | None]
    started_at: Mapped[datetime | None]
    finished_at: Mapped[datetime | None]
    wall_s: Mapped[float | None]
    gpu_s: Mapped[float | None]
    peak_mem_mb: Mapped[float | None]
    cost_usd: Mapped[float | None]
    outputs: Mapped[dict[str, Any]] = mapped_column(default=dict)
    summary: Mapped[dict[str, Any]] = mapped_column(default=dict)
    error: Mapped[str | None]
    job: Mapped[Job] = relationship(back_populates="stages")


class AssetVersion(Base):
    __tablename__ = "asset_versions"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_id)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), index=True)
    method: Mapped[str]  # Condition: A, B, C, D
    parent_id: Mapped[str | None] = mapped_column(ForeignKey("asset_versions.id"))
    glb_key: Mapped[str]
    lod_keys: Mapped[list[Any]] = mapped_column(default=list)
    triangles: Mapped[int | None]
    bytes: Mapped[int | None]
    unverified_surface_share: Mapped[float | None]
    provenance: Mapped[dict[str, Any]] = mapped_column(default=dict)
    created_at: Mapped[datetime] = mapped_column(default=_now)
    scores: Mapped[list[Score]] = relationship(back_populates="asset")


class Score(Base):
    __tablename__ = "scores"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_id)
    asset_id: Mapped[str] = mapped_column(ForeignKey("asset_versions.id"), index=True)
    photo_id: Mapped[str] = mapped_column(ForeignKey("photos.id"))
    fit_iou: Mapped[float]
    fit_ambiguous: Mapped[bool]
    metrics: Mapped[dict[str, Any]]  # name -> value; only metrics actually computed
    passed: Mapped[bool | None]  # None when the fit was not trusted enough to score
    failing_regions: Mapped[list[Any]] = mapped_column(default=list)
    render_keys: Mapped[dict[str, Any]] = mapped_column(default=dict)
    asset: Mapped[AssetVersion] = relationship(back_populates="scores")


class RepairAction(Base):
    __tablename__ = "repair_actions"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_id)
    asset_in_id: Mapped[str] = mapped_column(ForeignKey("asset_versions.id"))
    asset_out_id: Mapped[str | None] = mapped_column(ForeignKey("asset_versions.id"))
    rung: Mapped[int]
    region: Mapped[dict[str, Any]]
    tool: Mapped[str]
    before: Mapped[dict[str, Any]]
    after: Mapped[dict[str, Any]]
    kept: Mapped[bool]


class DeviceRun(Base):
    __tablename__ = "device_runs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_id)
    asset_id: Mapped[str] = mapped_column(ForeignKey("asset_versions.id"))
    phone: Mapped[str]
    network: Mapped[str]
    first_view_ms: Mapped[float | None]
    full_quality_ms: Mapped[float | None]
    frame_ms_median: Mapped[float | None]
    frame_ms_p90: Mapped[float | None]
    peak_memory_mb: Mapped[float | None]
    bytes_downloaded: Mapped[int | None]
    crashed: Mapped[bool]


def make_engine(url: str) -> Engine:
    kwargs: dict[str, Any] = {}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        if ":memory:" in url or url in ("sqlite://", "sqlite+pysqlite://"):
            from sqlalchemy.pool import StaticPool

            kwargs["poolclass"] = StaticPool
    else:
        # Hosted Postgres (Neon, Supabase) drops idle connections; check before each use.
        kwargs["pool_pre_ping"] = True
    return create_engine(url, **kwargs)


def make_sessionmaker(engine: Engine) -> sessionmaker[Any]:
    return sessionmaker(engine, expire_on_commit=False)
