"""GPU worker: ``python -m proof3d_worker_gpu [--drain]``. Runs on Modal, Kaggle or any GPU box
(ADR 0006, docs/gpu.md). ``--drain`` processes the queue and exits.

Stages owned here (PRD 4.1): ingest (BiRefNet masks, view labels), generate_shape
(Hunyuan3D-2mv, and Hunyuan3D-2 as the single-photo baseline), project_texture, verify (camera fit
via proof3d_geometry + nvdiffrast, LPIPS / DINOv2 / FLIP) and the repair controller.
Handlers are registered as they are built; see docs/architecture.md#build-status. Ingest
masks photos (transparency or BiRefNet); generate_shape calls the Hunyuan3D container on
Modal (ADR 0009); verify scores silhouettes.
"""

from __future__ import annotations

import logging

from proof3d_api.worker import Handler, run_worker
from proof3d_pipeline import StageName, WorkerKind
from proof3d_worker_gpu.generate import generate_shape
from proof3d_worker_gpu.ingest import ingest
from proof3d_worker_gpu.verify import verify

HANDLERS: dict[StageName, Handler] = {
    StageName.INGEST: ingest,
    StageName.GENERATE_SHAPE: generate_shape,
    StageName.VERIFY: verify,
}


def main() -> None:  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    run_worker(WorkerKind.GPU, HANDLERS)


if __name__ == "__main__":  # pragma: no cover
    main()
