"""Ingest and verify end to end: upload, import, drain the GPU worker, read the scorecard.

The product is synthetic (proof3d_geometry.synthetic): these tests show the stages work, not
how accurate any generator is.
"""

from __future__ import annotations

import io
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import trimesh
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy.orm import sessionmaker

from proof3d_api.db import Job, Product, make_engine, make_sessionmaker
from proof3d_api.main import create_app
from proof3d_api.settings import Settings
from proof3d_api.worker import Handler, drain
from proof3d_geometry.synthetic import photo_mask, tripod_mesh
from proof3d_pipeline import StageMetrics, StageName, StageResult, StageStatus, WorkerKind
from proof3d_pipeline.storage import LocalStorage
from proof3d_worker_gpu.__main__ import HANDLERS
from proof3d_worker_gpu.ingest import ingest, mask_from_image

# Fewer, coarser fit evaluations than production, so the suite stays fast.
CHEAP = {
    "verify": {
        "camera_fit": {
            "resolution": 64,
            "supersample": 1,
            "top_k": 2,
            "max_iter": 80,
            "polish_restarts": 0,
        }
    }
}
VIEWS = {"front.png": (0.0, 10.0), "left.png": (90.0, 10.0), "top.png": (30.0, 60.0)}


def png(mask: np.ndarray, background: bool = False) -> bytes:
    """A background-removed product photo: grey product, transparent elsewhere."""
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[..., :3] = np.where(mask[..., None], 170, 240)
    rgba[..., 3] = 255 if background else mask * 255
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    return buf.getvalue()


def jpeg(mask: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.where(mask, 170, 240).astype(np.uint8)).save(buf, format="JPEG")
    return buf.getvalue()


def glb(mesh: trimesh.Trimesh) -> bytes:
    data: bytes = mesh.export(file_type="glb")
    return data


@dataclass
class Env:
    client: TestClient
    sessions: sessionmaker[Any]
    storage: LocalStorage

    def upload(self, name: str, data: bytes) -> str:
        r = self.client.post("/uploads", params={"filename": name}, content=data)
        assert r.status_code == 201, r.text
        key: str = r.json()["key"]
        return key

    def drain(self, handlers: dict[StageName, Handler] | None = None) -> int:
        return drain(self.sessions, WorkerKind.GPU, handlers or HANDLERS, self.storage, 3)

    def stages(self, job_id: str) -> dict[str, dict[str, Any]]:
        return {s["stage"]: s for s in self.client.get(f"/jobs/{job_id}").json()["stages"]}


@pytest.fixture
def env(tmp_path: Path) -> Iterator[Env]:
    url = f"sqlite+pysqlite:///{tmp_path / 'db.sqlite'}"
    storage = LocalStorage(tmp_path / "store")
    settings = Settings(database_url=url, _env_file=None)  # type: ignore[call-arg]
    with TestClient(create_app(settings, create_tables=True, storage=storage)) as client:
        yield Env(client, make_sessionmaker(make_engine(url)), storage)


@pytest.fixture(scope="module")
def product() -> trimesh.Trimesh:
    return tripod_mesh()


def _import_model(
    env: Env, photos: dict[str, bytes], model: bytes, held_out: list[str]
) -> dict[str, Any]:
    keys = {name: env.upload(name, data) for name, data in photos.items()}
    body = {
        "photos": list(keys.values()),
        "title": "Synthetic tripod",
        "config": CHEAP,
        "model": {
            "glb": env.upload("model.glb", model),
            "generator": "test",
            "held_out": [keys[n] for n in held_out],
        },
    }
    r = env.client.post("/products/import", json=body)
    assert r.status_code == 201, r.text
    out: dict[str, Any] = r.json()
    return out


def _photos(mesh: trimesh.Trimesh) -> dict[str, bytes]:
    return {name: png(photo_mask(mesh, az, el)) for name, (az, el) in VIEWS.items()}


def test_correct_model_agrees_with_held_out_photos(env: Env, product: trimesh.Trimesh) -> None:
    ids = _import_model(env, _photos(product), glb(product), ["left.png", "top.png"])

    assert env.drain() == 2  # ingest, verify; generation, texturing and repair are skipped
    stages = env.stages(ids["job_id"])
    assert stages["ingest"]["status"] == "succeeded"
    assert stages["ingest"]["summary"]["held_out"] == ["left.png", "top.png"]
    assert stages["verify"]["status"] == "succeeded", stages["verify"]["error"]
    assert stages["optimize"]["status"] == "queued"  # handed on to the CPU worker

    card = env.client.get(f"/assets/{ids['asset_id']}/scorecard").json()
    assert card["verdict"] == "silhouette_agrees"
    assert sorted(s["photo_name"] for s in card["scores"]) == ["left.png", "top.png"]
    assert all(s["fit_iou"] > 0.9 and s["passed"] is None for s in card["scores"])
    ev = card["evidence"]
    assert 0 < ev["checkable_share"] <= ev["evidenced_share"] <= 1
    assert ev["invented_share"] == pytest.approx(1 - ev["evidenced_share"])
    assert card["unverified_surface_share"] == pytest.approx(1 - ev["checkable_share"])
    assert [f["name"] for f in ev["input_fits"]] == ["front.png"]
    assert ev["next_photos"]
    assert ev["next_photos"][0]["gain"] > 0
    report = json.loads(env.storage.get(ev["report_key"]))
    assert [p["name"] for p in report["photos"]] == ["front.png", "left.png", "top.png"]


def test_wrong_depth_is_caught_by_held_out_photos(env: Env, product: trimesh.Trimesh) -> None:
    wrong = tripod_mesh(depth_arm=0.9)  # the arm pointing at the front camera is too long
    ids = _import_model(env, _photos(product), glb(wrong), ["left.png", "top.png"])
    env.drain()

    card = env.client.get(f"/assets/{ids['asset_id']}/scorecard").json()
    assert card["verdict"] == "disagrees"
    assert any(s["passed"] is False for s in card["scores"])
    assert any("disagrees with held-out photo" in n for n in card["evidence"]["notes"])
    assert card["evidence"]["next_photos"] == []  # fix the model first; more photos won't help
    verify = env.stages(ids["job_id"])["verify"]["summary"]
    assert verify["results"][0]["verdict"] == "disagrees"
    assert verify["not_scored_yet"] == ["lpips", "dino", "flip", "edges"]


def test_rejected_held_out_photo_leaves_nothing_checkable(
    env: Env, product: trimesh.Trimesh
) -> None:
    photos = {
        "front.png": png(photo_mask(product, 0, 10)),
        "left.jpg": jpeg(photo_mask(product, 90, 10)),  # no transparency: cannot be masked yet
    }
    ids = _import_model(env, photos, glb(product), ["left.jpg"])
    env.drain()

    ingest_summary = env.stages(ids["job_id"])["ingest"]["summary"]
    assert ingest_summary["rejected"][0]["photo"] == "left.jpg"
    assert ingest_summary["rejected"][0]["role"] == "held_out"
    assert "background removal is not built yet" in ingest_summary["rejected"][0]["reasons"][0]
    card = env.client.get(f"/assets/{ids['asset_id']}/scorecard").json()
    assert card["verdict"] == "not_checkable"
    assert card["evidence"]["checkable_share"] == 0.0
    assert any("No held-out photo" in n for n in card["evidence"]["notes"])


def test_rerunning_verify_replaces_scores(env: Env, product: trimesh.Trimesh) -> None:
    ids = _import_model(env, _photos(product), glb(product), ["left.png"])
    env.drain()
    assert env.client.post(f"/jobs/{ids['job_id']}/rerun", params={"from": "verify"}).is_success
    assert env.drain() == 1
    card = env.client.get(f"/assets/{ids['asset_id']}/scorecard").json()
    assert len(card["scores"]) == 1


def test_own_pipeline_import_holds_out_the_last_photo(env: Env, product: trimesh.Trimesh) -> None:
    keys = [env.upload(n, d) for n, d in _photos(product).items()]
    job_id = env.client.post("/products/import", json={"photos": keys}).json()["job_id"]
    # Ingest succeeds; generate_shape has no handler yet, so it fails as "not built".
    assert env.drain({StageName.INGEST: ingest}) == 2
    with env.sessions() as s:
        job = s.get(Job, job_id)
        assert job is not None
        product_row = s.get(Product, job.product_id)
        assert product_row is not None
        roles = {p.source.rsplit("/", 1)[1]: p.role for p in product_row.photos}
        assert all(p.mask_key for p in product_row.photos)
    assert roles == {"front.png": "input", "left.png": "input", "top.png": "held_out"}
    assert "not built yet" in env.stages(job_id)["generate_shape"]["error"]


def test_no_usable_photos_fails_with_reasons(env: Env, product: trimesh.Trimesh) -> None:
    key = env.upload("front.jpg", jpeg(photo_mask(product, 0, 10)))
    job_id = env.client.post("/products/import", json={"photos": [key]}).json()["job_id"]
    env.drain()
    error = env.stages(job_id)["ingest"]["error"]
    assert error.startswith("No usable photos. front.jpg:")
    assert "upload a background-removed PNG" in error


def test_url_photos_are_not_fetched_yet(env: Env) -> None:
    job_id = env.client.post(
        "/products/import", json={"photos": ["https://example.com/front.png"]}
    ).json()["job_id"]
    env.drain()
    assert "fetching photos from URLs is not built yet" in env.stages(job_id)["ingest"]["error"]


def _ok(stage: StageName) -> Handler:
    def handler(ctx: Any) -> StageResult:
        return StageResult(
            stage=stage, status=StageStatus.SUCCEEDED, metrics=StageMetrics(wall_s=0)
        )

    return handler


def test_verify_without_a_model_says_so(env: Env, product: trimesh.Trimesh) -> None:
    keys = [env.upload(n, d) for n, d in _photos(product).items()]
    job_id = env.client.post("/products/import", json={"photos": keys}).json()["job_id"]
    handlers = {
        **HANDLERS,
        StageName.GENERATE_SHAPE: _ok(StageName.GENERATE_SHAPE),
        StageName.PROJECT_TEXTURE: _ok(StageName.PROJECT_TEXTURE),
    }
    env.drain(handlers)
    assert env.stages(job_id)["verify"]["error"].startswith("Nothing to verify")


def test_unreadable_model_fails_verify(env: Env, product: trimesh.Trimesh) -> None:
    ids = _import_model(env, _photos(product), b"glTF" + b"\0" * 64, ["left.png"])
    env.drain()
    error = env.stages(ids["job_id"])["verify"]["error"]
    assert error == "The model file could not be read as GLB."


def _rgba(alpha: np.ndarray) -> bytes:
    rgba = np.zeros((*alpha.shape, 4), dtype=np.uint8)
    rgba[..., 3] = alpha
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    return buf.getvalue()


def test_mask_flags() -> None:
    blank = np.zeros((40, 40), dtype=np.uint8)
    centred = blank.copy()
    centred[10:30, 10:30] = 255
    mask, flags = mask_from_image(_rgba(centred))
    assert mask is not None
    assert mask.sum() == 400
    assert flags == []

    cropped = blank.copy()
    cropped[0:20, 10:30] = 255
    assert mask_from_image(_rgba(cropped))[1] == ["touches_border"]

    two = blank.copy()
    two[5:15, 5:15] = 255
    two[25:35, 25:35] = 255
    assert mask_from_image(_rgba(two))[1] == ["multiple_objects"]

    speck = centred.copy()
    speck[2, 2] = 255  # a stray pixel is not a second object
    assert mask_from_image(_rgba(speck))[1] == []

    assert mask_from_image(_rgba(blank)) == (None, ["empty_mask"])
    assert mask_from_image(_rgba(blank + 255)) == (None, ["needs_background_removal"])
    assert mask_from_image(b"not an image") == (None, ["unreadable"])

    palette = Image.fromarray(centred // 255, mode="L").convert("P")
    buf = io.BytesIO()
    palette.save(buf, format="PNG", transparency=0)
    mask, flags = mask_from_image(buf.getvalue())
    assert mask is not None
    assert mask.sum() == 400
