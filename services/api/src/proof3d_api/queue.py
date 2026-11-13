"""Postgres-backed stage queue (ADR 0002).

A job is a chain of StageRun rows. Exactly one stage per job is QUEUED or RUNNING at a time;
completing it queues the next. Workers claim with ``FOR UPDATE SKIP LOCKED`` so several workers
of the same kind never take the same row. Failures retry up to ``max_attempts``.
"""

from __future__ import annotations

from collections.abc import Collection
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


def create_stage_runs(job: Job, skip: Collection[StageName] = ()) -> None:
    """Create the job's stage chain. Stages in ``skip`` are not part of this job (an imported
    model has no generation stage) and are passed over when the queue advances."""
    for i, stage in enumerate(STAGE_ORDER):
        planned = stage in skip
        run = StageRun(
            stage=stage.value,
            position=i,
            worker=worker_for(stage).value,
            status=(StageStatus.SKIPPED if planned else StageStatus.PENDING).value,
            planned_skip=planned,
        )
        job.stages.append(run)
    first = _next_runnable(job, -1)
    if first is not None:
        _queue(first)


def _next_runnable(job: Job, after: int) -> StageRun | None:
    later = sorted(
        (s for s in job.stages if s.position > after and not s.planned_skip),
        key=lambda s: s.position,
    )
    return later[0] if later else None


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
        nxt = _next_runnable(run.job, run.position)
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
    first = next(s for s in job.stages if s.stage == stage.value)
    if first.planned_skip:
        raise RerunConflictError(f"{stage.value} is not part of this job")
    earlier = [s for s in job.stages if s.position < first.position]
    done = {StageStatus.SUCCEEDED.value, StageStatus.SKIPPED.value}
    if any(s.status not in done for s in earlier):
        raise RerunConflictError(
            f"cannot rerun from {stage.value}: an earlier stage has not succeeded"
        )
    # Checks first, then changes: a refused rerun leaves the job untouched.
    reset = {s.value for s in stages_from(stage)}
    for run in job.stages:
        if run.stage in reset and not run.planned_skip:
            run.status = StageStatus.PENDING.value
            run.attempts = 0
            run.error = None
            run.started_at = run.finished_at = run.queued_at = None
    _queue(first)
    session.flush()
