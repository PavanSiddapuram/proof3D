"""Run the GPU worker on Modal: process every queued GPU stage, then exit (ADR 0006).

    make gpu-drain                 # or: uv run --with modal==1.6.3 modal run <this file>
    make gpu-drain ARGS="--max-stages 1"

Modal bills per second while the container runs, so the function drains the queue and stops
instead of polling. It reads its configuration from a Modal secret named ``proof3d``; see
docs/gpu.md for the keys. The database and storage must be reachable from Modal, so a localhost
database or local-disk storage is refused before any GPU time is spent on stages.

The image installs exactly the third-party versions in uv.lock (exported at build time on your
machine) plus the four workspace packages copied from source, so Modal runs the code CI tests.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS = ROOT / ".data" / "modal" / "requirements.txt"
GPU = os.environ.get("PROOF3D_MODAL_GPU", "L40S")
UV_VERSION = "0.11.26"


def _export_requirements() -> None:
    REQUIREMENTS.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "uv",
            "export",
            "--frozen",
            "--no-dev",
            "--package",
            "proof3d-worker-gpu",
            "--no-emit-workspace",
            "--no-hashes",
            "--quiet",
            "--output-file",
            str(REQUIREMENTS),
        ],
        check=True,
        cwd=ROOT,
    )


if modal.is_local():
    _export_requirements()

# Workspace packages are added by path, not by import: the dev environment deliberately does not
# install the GPU worker (it would pull torch onto every laptop). /root is on the container's
# sys.path.
SOURCES = {
    "proof3d_api": "services/api/src/proof3d_api",
    "proof3d_pipeline": "packages/pipeline/src/proof3d_pipeline",
    "proof3d_geometry": "packages/geometry/src/proof3d_geometry",
    "proof3d_worker_gpu": "services/worker-gpu/src/proof3d_worker_gpu",
}

image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    requirements=[str(REQUIREMENTS)], uv_version=UV_VERSION
)
for package, path in SOURCES.items():
    image = image.add_local_dir(ROOT / path, f"/root/{package}", ignore=["**/__pycache__"])

app = modal.App("proof3d-worker-gpu", image=image)


@app.function(
    gpu=GPU,
    secrets=[
        modal.Secret.from_name(
            "proof3d", required_keys=["PROOF3D_DATABASE_URL", "PROOF3D_STORAGE_BACKEND"]
        )
    ],
    timeout=60 * 60,
)
def drain(max_stages: int | None = None) -> int:
    import logging

    from proof3d_api.settings import get_settings
    from proof3d_api.worker import run_worker
    from proof3d_pipeline import WorkerKind
    from proof3d_worker_gpu.__main__ import HANDLERS

    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    if settings.storage_backend != "s3":
        raise RuntimeError("On Modal, set PROOF3D_STORAGE_BACKEND=s3; local disk is discarded.")
    if any(host in settings.database_url for host in ("@localhost", "@127.0.0.1", "@db:")):
        raise RuntimeError("PROOF3D_DATABASE_URL points at a local database Modal cannot reach.")
    argv = ["--drain"] + ([] if max_stages is None else ["--max", str(max_stages)])
    return run_worker(WorkerKind.GPU, HANDLERS, argv, settings)


@app.local_entrypoint()
def main(max_stages: int = 0) -> None:
    processed = drain.remote(max_stages or None)
    print(f"Modal GPU worker processed {processed} stage(s) on {GPU}.")
