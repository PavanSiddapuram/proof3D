from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class ExternalModel(BaseModel):
    """A model another generator already made, imported to be verified, not regenerated."""

    model_config = ConfigDict(extra="forbid")

    glb: str = Field(min_length=1, description="URL or storage key of the GLB")
    generator: str = Field(
        min_length=1, max_length=64, description="Who made it, e.g. versa_ai, tripo, rodin"
    )
    generator_version: str | None = Field(default=None, max_length=64)
    held_out: list[str] = Field(
        min_length=1,
        description=(
            "Photos, from `photos`, that the generator never saw. Only these can verify the "
            "model; photos it was built from can only show it is consistent with its input."
        ),
    )


class ImportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_url: HttpUrl | None = Field(
        default=None, description="Product page on your own demo store (PRD non-goals)"
    )
    photos: list[str] = Field(default_factory=list, description="Image URLs or storage keys")
    title: str | None = None
    category: str | None = None
    config: dict[str, Any] = Field(default_factory=dict, description="PipelineConfig overrides")
    model: ExternalModel | None = Field(
        default=None, description="Verify this existing model instead of generating one"
    )

    @model_validator(mode="after")
    def _needs_input(self) -> ImportRequest:
        if self.source_url is None and not self.photos:
            raise ValueError("provide a product URL or at least one photo")
        if self.model is not None:
            unknown = sorted(set(self.model.held_out) - set(self.photos))
            if unknown:
                raise ValueError(f"held_out photos must be listed in photos: {unknown}")
        return self


class UploadOut(BaseModel):
    key: str = Field(description="Storage key; pass it in `photos` or `model.glb` on import")
    bytes: int
    kind: Literal["image", "model"]


class ImportResponse(BaseModel):
    job_id: str
    product_id: str
    asset_id: str | None = Field(
        default=None, description="The imported model's asset version, when one was given"
    )


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
    photo_name: str | None = None
    fit_iou: float
    fit_ambiguous: bool
    metrics: dict[str, Any]
    passed: bool | None
    failing_regions: list[Any]
    render_keys: dict[str, Any]


class NextPhotoOut(BaseModel):
    view: str = Field(description="front, back, left, right, three_quarter_*, top, underside")
    azimuth_deg: float
    elevation_deg: float
    gain: float = Field(description="Share of the surface this photo would newly show")
    covered_after: float = Field(description="Share shown by some photo once it is taken")


class InputFitOut(BaseModel):
    photo_id: str
    name: str
    fit_iou: float
    low_iou: bool
    ambiguous: bool
    surface_shown: float


class EvidenceOut(BaseModel):
    """Which parts of the surface any photo shows (ADR 0007)."""

    checkable_share: float = Field(description="Shown by a held-out photo: the score covers it")
    evidenced_share: float = Field(description="Shown by any photo, input or held-out")
    invented_share: float = Field(description="Shown by no photo: the generator guessed it")
    faces_checked: int
    next_photos: list[NextPhotoOut]
    input_fits: list[InputFitOut] = Field(
        description="How well the model matches the photos it was built from (consistency only)"
    )
    notes: list[str]
    report_key: str


Verdict = Literal["disagrees", "silhouette_agrees", "not_checkable"]


class ScorecardOut(BaseModel):
    asset_id: str
    method: str
    verdict: Verdict | None = Field(
        description=(
            "disagrees: a held-out photo's silhouette does not match. silhouette_agrees: every "
            "held-out silhouette matches; appearance is not scored yet. not_checkable: no "
            "usable held-out photo. None: not verified yet."
        )
    )
    unverified_surface_share: float | None
    evidence: EvidenceOut | None
    triangles: int | None
    bytes: int | None
    scores: list[ScoreOut]
    provenance: dict[str, Any]


class CompareOut(BaseModel):
    product_id: str
    conditions: dict[str, ScorecardOut]
