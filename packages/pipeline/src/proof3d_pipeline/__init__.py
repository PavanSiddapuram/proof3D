"""Typed stage interfaces, configs and results for the Proof3D pipeline (PRD section 5)."""

from proof3d_pipeline.config import PipelineConfig, config_hash, load_config
from proof3d_pipeline.models import (
    Condition,
    PhotoRole,
    Provenance,
    StageMetrics,
    StageResult,
    StageStatus,
    ViewLabel,
)
from proof3d_pipeline.stages import STAGE_ORDER, StageName, WorkerKind, stages_from, worker_for

__all__ = [
    "STAGE_ORDER",
    "Condition",
    "PhotoRole",
    "PipelineConfig",
    "Provenance",
    "StageMetrics",
    "StageName",
    "StageResult",
    "StageStatus",
    "ViewLabel",
    "WorkerKind",
    "config_hash",
    "load_config",
    "stages_from",
    "worker_for",
]
