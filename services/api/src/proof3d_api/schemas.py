from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class ImportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_url: HttpUrl | None = Field(
        default=None, description="Product page on your own demo store (PRD non-goals)"
    )
    photos: list[str] = Field(default_factory=list, description="Image URLs or storage keys")
    title: str | None = None
    category: str | None = None
    config: dict[str, Any] = Field(default_factory=dict, description="PipelineConfig overrides")

    @model_validator(mode="after")
    def _needs_input(self) -> ImportRequest:
        if self.source_url is None and not self.photos:
            raise ValueError("provide a product URL or at least one photo")
        return self


class ImportResponse(BaseModel):
    job_id: str
    product_id: str


class StageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    stage: str
    worker: str
    status: str
    attempts: int
    started_at: datetime | None
    finished_at: datetime | None
    wall_s: float | None
    gpu_s: float | None
    cost_usd: float | None
    summary: dict[str, Any]
    error: str | None


class JobOut(BaseModel):
    id: str
    product_id: str
    pipeline_version: str
    config_hash: str
    status: str
    stages: list[StageOut]
    total_wall_s: float
    total_gpu_s: float
    total_cost_usd: float | None


class ScoreOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    photo_id: str
    fit_iou: float
    fit_ambiguous: bool
    metrics: dict[str, Any]
    passed: bool | None
    failing_regions: list[Any]
    render_keys: dict[str, Any]


class ScorecardOut(BaseModel):
    asset_id: str
    method: str
    unverified_surface_share: float | None
    triangles: int | None
    bytes: int | None
    scores: list[ScoreOut]
    provenance: dict[str, Any]


class CompareOut(BaseModel):
    product_id: str
    conditions: dict[str, ScorecardOut]
