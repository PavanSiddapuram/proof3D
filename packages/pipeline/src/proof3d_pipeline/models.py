"""Shared typed values: statuses, stage results, provenance (standards section 7)."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from proof3d_pipeline.stages import StageName


class StageStatus(StrEnum):
    PENDING = "pending"  # waiting for an earlier stage
    QUEUED = "queued"  # ready; a worker may claim it
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class PhotoRole(StrEnum):
    INPUT = "input"
    HELD_OUT = "held_out"
    REJECTED = "rejected"


class ViewLabel(StrEnum):
    FRONT = "front"
    BACK = "back"
    LEFT = "left"
    RIGHT = "right"
    THREE_QUARTER_LEFT = "three_quarter_left"
    THREE_QUARTER_RIGHT = "three_quarter_right"
    TOP = "top"
    CLOSE_UP = "close_up"


class Condition(StrEnum):
    """Evaluation conditions compared on identical inputs (PRD 6)."""

    A_SINGLE_GENERATED = "A"
    B_MULTI_GENERATED = "B"
    C_MULTI_PROJECTED = "C"
    D_MULTI_PROJECTED_REPAIRED = "D"
    # A model made by another generator (VersaAI, Tripo, Rodin, ...) and imported for
    # verification only. Proof3D scores it; it does not generate or retexture it.
    EXTERNAL = "external"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class StageMetrics(_Frozen):
    """Every stage measures itself (standards 3, rule 5)."""

    wall_s: float = Field(ge=0)
    gpu_s: float = Field(default=0.0, ge=0)
    peak_mem_mb: float | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)


class Provenance(_Frozen):
    """Tags attached to every reported result so any number can be reproduced."""

    git_commit: str
    pipeline_version: str
    config_hash: str
    seed: int
    dataset_manifest_version: str | None = None
    models: dict[str, str] = Field(
        default_factory=dict, description="model name -> version@weight-sha256"
    )
    gpu: str | None = None
    driver: str | None = None


class StageResult(_Frozen):
    """What a stage returns. Outputs are storage keys, never in-memory blobs."""

    stage: StageName
    status: StageStatus
    metrics: StageMetrics
    outputs: dict[str, str] = Field(default_factory=dict, description="name -> storage key")
    summary: dict[str, Any] = Field(default_factory=dict)
    error: str | None = Field(
        default=None, description="Human-readable error shown in the UI, not a stack trace"
    )
