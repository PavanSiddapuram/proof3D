"""GPU worker: ``python -m proof3d_worker_gpu``. Runs on the rented 24 GB+ GPU.

Stages owned here (PRD 4.1): ingest (BiRefNet masks, view labels), generate_shape
(Hunyuan3D-2mv and the TRELLIS.2 single-photo baseline), project_texture, verify (camera fit
via proof3d_geometry + nvdiffrast, LPIPS / DINOv2 / FLIP) and the repair controller.
Handlers are registered as they are built; see docs/architecture.md#build-status.
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
        WorkerKind.GPU,
        HANDLERS,
        LocalStorage(settings.storage_root),
        settings.max_stage_attempts,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
