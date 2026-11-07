from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from sqlalchemy.orm import sessionmaker

from proof3d_api import queue
from proof3d_api.db import Base, Job, Product, StageRun, make_engine, make_sessionmaker
from proof3d_api.worker import StageContext, StageError, run_once
from proof3d_pipeline import StageMetrics, StageName, StageResult, StageStatus, WorkerKind
from proof3d_pipeline.storage import LocalStorage


@pytest.fixture
def sessions(tmp_path: Path) -> sessionmaker[Any]:
    engine = make_engine(f"sqlite+pysqlite:///{tmp_path / 'q.sqlite'}")
    Base.metadata.create_all(engine)
    return make_sessionmaker(engine)


@pytest.fixture
def job_id(sessions: sessionmaker[Any]) -> str:
    with sessions() as s, s.begin():
        job = Job(product=Product(title="Mug"), pipeline_version="t", config={}, config_hash="h")
        queue.create_stage_runs(job)
        s.add(job)
        s.flush()
        return job.id


def _stage(sessions: sessionmaker[Any], job_id: str, stage: StageName) -> StageRun:
    with sessions() as s:
        return next(r for r in s.get(Job, job_id).stages if r.stage == stage.value)  # type: ignore[union-attr]


def test_handler_result_recorded_and_next_stage_queued(
    sessions: sessionmaker[Any], job_id: str, tmp_path: Path
) -> None:
    def ingest(ctx: StageContext) -> StageResult:
        key = ctx.storage.put(f"jobs/{ctx.job_id}/ingest.json", b"{}")
        return StageResult(
            stage=ctx.stage,
            status=StageStatus.SUCCEEDED,
            metrics=StageMetrics(wall_s=0.2, gpu_s=0.2),
            outputs={"manifest": key},
            summary={"usable_photos": 6},
        )

    storage = LocalStorage(tmp_path / "store")
    handlers = {StageName.INGEST: ingest}
    assert run_once(sessions, WorkerKind.GPU, handlers, storage, max_attempts=3)
    run = _stage(sessions, job_id, StageName.INGEST)
    assert run.status == "succeeded"
    assert run.summary == {"usable_photos": 6}
    assert storage.exists(run.outputs["manifest"])
    assert _stage(sessions, job_id, StageName.GENERATE_SHAPE).status == "queued"


def test_unbuilt_stage_fails_clearly_without_retry(
    sessions: sessionmaker[Any], job_id: str, tmp_path: Path
) -> None:
    assert run_once(sessions, WorkerKind.GPU, {}, LocalStorage(tmp_path), max_attempts=3)
    run = _stage(sessions, job_id, StageName.INGEST)
    assert run.status == "failed"
    assert run.error == "The ingest stage is not built yet."
    assert not run_once(sessions, WorkerKind.GPU, {}, LocalStorage(tmp_path), max_attempts=3)


def test_user_facing_error_message_kept(
    sessions: sessionmaker[Any], job_id: str, tmp_path: Path
) -> None:
    def ingest(ctx: StageContext) -> StageResult:
        raise StageError("No usable back view found; 31% of the surface is unverified")

    run_once(sessions, WorkerKind.GPU, {StageName.INGEST: ingest}, LocalStorage(tmp_path), 3)
    run = _stage(sessions, job_id, StageName.INGEST)
    assert run.status == "failed"
    assert run.error is not None
    assert run.error.startswith("No usable back view")


def test_crash_is_retried_then_fails(
    sessions: sessionmaker[Any], job_id: str, tmp_path: Path
) -> None:
    def ingest(ctx: StageContext) -> StageResult:
        raise RuntimeError("CUDA out of memory")

    handlers = {StageName.INGEST: ingest}
    for _ in range(3):
        run_once(sessions, WorkerKind.GPU, handlers, LocalStorage(tmp_path), max_attempts=3)
    run = _stage(sessions, job_id, StageName.INGEST)
    assert run.status == "failed"
    assert run.attempts == 3
    assert run.error == "The ingest stage crashed; see the job trace."
