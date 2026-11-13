"""End-to-end demo of verifying an imported model, through the real API, queue and database.

    uv run python -m proof3d_api.migrate                        # once per database
    uv run uvicorn proof3d_api.main:create_app --factory        # terminal 1
    uv run python eval/scripts/demo_verify.py [--wrong]         # terminal 2

It makes a SYNTHETIC product (proof3d_geometry.synthetic), writes its photos and a model to
.data/demo/, then:

1. uploads them with ``POST /uploads``;
2. imports the model with ``POST /products/import``, holding back the left and top photos;
3. runs the GPU worker's queue in this process (ingest, verify);
4. prints the job's stages and the scorecard from the API.

``--wrong`` imports a model whose arm pointing at the front camera is too long: it still
matches the front photo, and the held-out photos should catch it. Numbers from this demo show
the pipeline works; they are not results about any generator.

Run it from the repository root, so the API and this script share ``.data/storage``.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from proof3d_geometry.synthetic import photo_mask, tripod_mesh

ROOT = Path(__file__).resolve().parents[2]
VIEWS = {"front.png": (0.0, 10.0), "left.png": (90.0, 10.0), "top.png": (30.0, 60.0)}
HELD_OUT = ["left.png", "top.png"]


def _request(method: str, url: str, body: bytes | None = None, json_body: Any = None) -> Any:
    headers = {}
    if json_body is not None:
        body = json.dumps(json_body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"{method} {url} -> {exc.code}: {exc.read().decode()}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"Cannot reach the API at {url} ({exc.reason}). Start it first:\n"
            "  uv run uvicorn proof3d_api.main:create_app --factory"
        ) from exc


def _png(mask: np.ndarray) -> bytes:
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[..., :3] = 170
    rgba[..., 3] = mask * 255
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    return buf.getvalue()


def make_files(out: Path, wrong: bool) -> dict[str, bytes]:
    true_product = tripod_mesh()
    model = tripod_mesh(depth_arm=0.9) if wrong else true_product
    files = {
        name: _png(photo_mask(true_product, az, el, width=480, height=360))
        for name, (az, el) in VIEWS.items()
    }
    files["model.glb"] = model.export(file_type="glb")
    out.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (out / name).write_bytes(data)
    return files


def drain_gpu_worker() -> int:
    # The GPU worker is not installed in the dev environment (it would pull in torch), so it
    # is imported from source, as in tests.
    sys.path.insert(0, str(ROOT / "services" / "worker-gpu" / "src"))
    from proof3d_api.worker import run_worker
    from proof3d_pipeline import WorkerKind
    from proof3d_worker_gpu.__main__ import HANDLERS

    return run_worker(WorkerKind.GPU, HANDLERS, ["--drain"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--wrong", action="store_true", help="import a model with wrong depth")
    parser.add_argument("--no-worker", action="store_true", help="leave the queue to a worker")
    parser.add_argument("--out", type=Path, default=ROOT / ".data" / "demo")
    args = parser.parse_args(argv)
    api = args.api.rstrip("/")

    files = make_files(args.out, args.wrong)
    print(f"Synthetic product written to {args.out}")
    keys = {
        name: _request("POST", f"{api}/uploads?filename={name}", body=data)["key"]
        for name, data in files.items()
    }
    photos = [keys[name] for name in VIEWS]
    ids = _request(
        "POST",
        f"{api}/products/import",
        json_body={
            "title": "Synthetic tripod (demo)",
            "photos": photos,
            "model": {
                "glb": keys["model.glb"],
                "generator": "demo-wrong-depth" if args.wrong else "demo-exact",
                "held_out": [keys[n] for n in HELD_OUT],
            },
        },
    )
    print(f"Imported: job {ids['job_id']}, asset {ids['asset_id']}")

    if not args.no_worker:
        t0 = time.perf_counter()
        drain_gpu_worker()
        print(f"GPU worker finished in {time.perf_counter() - t0:.1f} s")

    job = _request("GET", f"{api}/jobs/{ids['job_id']}")
    print(f"\nJob {job['status']}:")
    for s in job["stages"]:
        took = f"{s['wall_s']:.1f} s" if s["wall_s"] is not None else ""
        print(f"  {s['stage']:<16} {s['status']:<10} {took:>8}  {s['error'] or ''}")

    card = _request("GET", f"{api}/assets/{ids['asset_id']}/scorecard")
    print(f"\nVerdict: {card['verdict']}")
    for score in card["scores"]:
        flag = "  DISAGREES" if score["passed"] is False else ""
        print(f"  held-out {score['photo_name']}: silhouette IoU {score['fit_iou']:.3f}{flag}")
    ev = card["evidence"]
    if ev:
        for fit in ev["input_fits"]:
            print(f"  input    {fit['name']}: silhouette IoU {fit['fit_iou']:.3f}")
        print(f"Checkable by held-out photos: {ev['checkable_share']:.1%} of the surface")
        print(f"Shown by any photo:           {ev['evidenced_share']:.1%}")
        print(f"Shown by no photo (invented): {ev['invented_share']:.1%}")
        for p in ev["next_photos"]:
            where = f"azimuth {p['azimuth_deg']:.0f}°, elevation {p['elevation_deg']:.0f}°"
            print(f"  Next photo: {p['view']} ({where}) -> shows {p['gain']:.1%} more")
        for note in ev["notes"]:
            print(f"Note: {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
