"""CPU worker: ``python -m proof3d_worker_cpu``.

Stages owned here (PRD 4.1): optimise (LODs, KTX2, meshopt, FLIP gate) and publish (Shopify).
Repair rungs 1-2 run here too, called by the repair controller. Handlers are registered as
they are built; see docs/architecture.md#build-status.
"""

from __future__ import annotations

import logging

from proof3d_api.db import Base, make_engine, make_sessionmaker
from proof3d_api.settings import get_settings
from proof3d_api.worker import Handler, serve
from proof3d_pipeline import StageName, WorkerKind
from proof3d_pipeline.storage import LocalStorage

HANDLERS: dict[StageName, Handler] = {}


def main() -> None:  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    engine = make_engine(settings.database_url)
    if settings.create_tables:
        Base.metadata.create_all(engine)
    sessions = make_sessionmaker(engine)
    serve(
        sessions,
        WorkerKind.CPU,
        HANDLERS,
        LocalStorage(settings.storage_root),
        settings.max_stage_attempts,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
