"""GPU worker: ``python -m proof3d_worker_gpu [--drain]``. Runs on Modal, Kaggle or any GPU box
(ADR 0006, docs/gpu.md). ``--drain`` processes the queue and exits.

Stages owned here (PRD 4.1): ingest (BiRefNet masks, view labels), generate_shape
(Hunyuan3D-2mv and the TRELLIS.2 single-photo baseline), project_texture, verify (camera fit
via proof3d_geometry + nvdiffrast, LPIPS / DINOv2 / FLIP) and the repair controller.
Handlers are registered as they are built; see docs/architecture.md#build-status.
"""

from __future__ import annotations

import logging

from proof3d_api.worker import Handler, run_worker
from proof3d_pipeline import StageName, WorkerKind

HANDLERS: dict[StageName, Handler] = {}


def main() -> None:  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    run_worker(WorkerKind.GPU, HANDLERS)


if __name__ == "__main__":  # pragma: no cover
    main()
