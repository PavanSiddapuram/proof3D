"""CPU worker: ``python -m proof3d_worker_cpu [--drain]``.

Stages owned here (PRD 4.1): optimise (LODs, KTX2, meshopt, FLIP gate) and publish (Shopify).
Repair rungs 1-2 run here too, called by the repair controller. Handlers are registered as
they are built; see docs/architecture.md#build-status.
"""

from __future__ import annotations

import logging

from proof3d_api.worker import Handler, run_worker
from proof3d_pipeline import StageName, WorkerKind

HANDLERS: dict[StageName, Handler] = {}


def main() -> None:  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    run_worker(WorkerKind.CPU, HANDLERS)


if __name__ == "__main__":  # pragma: no cover
    main()
