"""Postgres-backed stage queue (ADR 0002).

A job is a chain of StageRun rows. Exactly one stage per job is QUEUED or RUNNING at a time;
completing it queues the next. Workers claim with ``FOR UPDATE SKIP LOCKED`` so several workers
of the same kind never take the same row. Failures retry up to ``max_attempts``.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from proof3d_api.db import Job, StageRun
from proof3d_pipeline import (
    STAGE_ORDER,
    StageName,
    StageResult,
    StageStatus,
    WorkerKind,
    stages_from,
    worker_for,
)


def _now() -> datetime:
    return datetime.now(UTC)


def create_stage_runs(job: Job) -> None:
    for i, stage in enumerate(STAGE_ORDER):
        job.stages.append(StageRun(stage=stage.value, position=i, worker=worker_for(stage).value))
    _queue(job.stages[0])


def _queue(run: StageRun) -> None:
    run.status = StageStatus.QUEUED.value
    run.queued_at = _now()


def claim(session: Session, worker: WorkerKind) -> StageRun | None:
    """Claim the oldest queued stage for this worker kind, or None."""
    stmt = (
        select(StageRun)
        .where(StageRun.worker == worker.value, StageRun.status == StageStatus.QUEUED.value)
        .order_by(StageRun.queued_at)
        .limit(1)
        .with_for_update(skip_locked=True)
    )
    run = session.scalars(stmt).first()
    if run is None:
        return None
    run.status = StageStatus.RUNNING.value
    run.attempts += 1
    run.started_at = _now()
    session.flush()
    return run


def complete(session: Session, run: StageRun, result: StageResult) -> None:
    run.status = result.status.value
    run.finished_at = _now()
    run.wall_s = result.metrics.wall_s
    run.gpu_s = result.metrics.gpu_s
    run.peak_mem_mb = result.metrics.peak_mem_mb
    run.cost_usd = result.metrics.cost_usd
    run.outputs = dict(result.outputs)
    run.summary = dict(result.summary)
    run.error = result.error
    if result.status in (StageStatus.SUCCEEDED, StageStatus.SKIPPED):
        nxt = next((s for s in run.job.stages if s.position == run.position + 1), None)
        if nxt is not None:
            _queue(nxt)
    session.flush()


def fail(session: Session, run: StageRun, message: str, max_attempts: int) -> None:
    """Record a failure. ``message`` is shown in the UI, so it must be human-readable."""
    run.error = message
    run.finished_at = _now()
    if run.attempts < max_attempts:
        _queue(run)
    else:
        run.status = StageStatus.FAILED.value
    session.flush()


class RerunConflictError(Exception):
    pass


def rerun_from(session: Session, job: Job, stage: StageName) -> None:
    """Reset ``stage`` and everything after it; earlier outputs stay valid because every stage
    is idempotent and writes to storage (PRD 4.1)."""
    if any(s.status == StageStatus.RUNNING.value for s in job.stages):
        raise RerunConflictError("a stage of this job is still running")
    reset = {s.value for s in stages_from(stage)}
    for run in job.stages:
        if run.stage in reset:
            run.status = StageStatus.PENDING.value
            run.attempts = 0
            run.error = None
            run.started_at = run.finished_at = run.queued_at = None
    first = next(s for s in job.stages if s.stage == stage.value)
    earlier = [s for s in job.stages if s.position < first.position]
    if any(s.status != StageStatus.SUCCEEDED.value for s in earlier):
        raise RerunConflictError(
            f"cannot rerun from {stage.value}: an earlier stage has not succeeded"
        )
    _queue(first)
    session.flush()
