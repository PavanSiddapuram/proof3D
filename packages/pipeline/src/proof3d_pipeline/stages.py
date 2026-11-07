"""Stage order and worker assignment (PRD 4.1 and 5)."""

from __future__ import annotations

from enum import StrEnum


class StageName(StrEnum):
    INGEST = "ingest"
    GENERATE_SHAPE = "generate_shape"
    PROJECT_TEXTURE = "project_texture"
    VERIFY = "verify"
    REPAIR = "repair"
    OPTIMIZE = "optimize"
    PHONE_TEST = "phone_test"
    PUBLISH = "publish"


class WorkerKind(StrEnum):
    GPU = "gpu"
    CPU = "cpu"
    HARNESS = "harness"


STAGE_ORDER: tuple[StageName, ...] = tuple(StageName)

# Repair spans both workers (rungs 1-2 are CPU, rungs 3-4 need the GPU); the repair
# controller itself runs on the GPU worker because every rung ends in a rescore.
_WORKER: dict[StageName, WorkerKind] = {
    StageName.INGEST: WorkerKind.GPU,
    StageName.GENERATE_SHAPE: WorkerKind.GPU,
    StageName.PROJECT_TEXTURE: WorkerKind.GPU,
    StageName.VERIFY: WorkerKind.GPU,
    StageName.REPAIR: WorkerKind.GPU,
    StageName.OPTIMIZE: WorkerKind.CPU,
    StageName.PHONE_TEST: WorkerKind.HARNESS,
    StageName.PUBLISH: WorkerKind.CPU,
}


def worker_for(stage: StageName) -> WorkerKind:
    return _WORKER[stage]


def stages_from(stage: StageName) -> tuple[StageName, ...]:
    """Stages to run when rerunning from ``stage`` (PRD 4.3 ``rerun?from=``)."""
    return STAGE_ORDER[STAGE_ORDER.index(stage) :]
