"""Worker loop shared by the CPU and GPU workers.

A worker owns a registry ``{StageName: handler}``. It claims a queued stage for its kind, runs
the handler, and records the typed result. A stage with no handler fails at once with a clear
message instead of retrying, so an unbuilt stage can never look like a slow one.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from proof3d_api import queue
from proof3d_api.db import Job, StageRun
from proof3d_pipeline import PipelineConfig, StageName, StageResult, WorkerKind
from proof3d_pipeline.storage import Storage

log = logging.getLogger("proof3d.worker")


@dataclass(frozen=True)
class StageContext:
    job_id: str
    product_id: str
    stage: StageName
    config: PipelineConfig
    storage: Storage
    session: Session
    attempt: int


Handler = Callable[[StageContext], StageResult]


class StageError(Exception):
    """Raised by handlers with a message fit for the UI ("No usable back view found ...")."""

    def __init__(self, message: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


def run_once(
    sessions: sessionmaker[Any],
    kind: WorkerKind,
    handlers: Mapping[StageName, Handler],
    storage: Storage,
    max_attempts: int,
) -> bool:
    """Claim and run one stage. Returns False when there was nothing to do."""
    with sessions() as session, session.begin():
        run = queue.claim(session, kind)
        if run is None:
            return False
        _execute(session, run, handlers, storage, max_attempts)
    return True


def _execute(
    session: Session,
    run: StageRun,
    handlers: Mapping[StageName, Handler],
    storage: Storage,
    max_attempts: int,
) -> None:
    stage = StageName(run.stage)
    job: Job = run.job
    handler = handlers.get(stage)
    if handler is None:
        queue.fail(session, run, f"The {stage.value} stage is not built yet.", max_attempts=0)
        return
    ctx = StageContext(
        job_id=job.id,
        product_id=job.product_id,
        stage=stage,
        config=PipelineConfig.model_validate(job.config),
        storage=storage,
        session=session,
        attempt=run.attempts,
    )
    log.info("stage start", extra={"job_id": job.id, "stage": stage.value})
    try:
        result = handler(ctx)
    except StageError as exc:
        queue.fail(session, run, str(exc), max_attempts if exc.retryable else 0)
        return
    except Exception:
        log.exception("stage crashed", extra={"job_id": job.id, "stage": stage.value})
        queue.fail(
            session, run, f"The {stage.value} stage crashed; see the job trace.", max_attempts
        )
        return
    queue.complete(session, run, result)


def serve(
    sessions: sessionmaker[Any],
    kind: WorkerKind,
    handlers: Mapping[StageName, Handler],
    storage: Storage,
    max_attempts: int,
    idle_sleep_s: float = 1.0,
) -> None:  # pragma: no cover - long-running loop
    log.info("worker ready", extra={"kind": kind.value, "stages": [s.value for s in handlers]})
    while True:
        if not run_once(sessions, kind, handlers, storage, max_attempts):
            time.sleep(idle_sleep_s)
