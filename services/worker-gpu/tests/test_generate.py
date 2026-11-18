"""generate_shape: conditions A and B from an import, through ingest, generation and verify.

The generator is a stand-in: B returns the true synthetic product, and A returns one with the
depth wrong, as a single photo cannot show depth. That shows the stages and the A-vs-B
comparison work end to end, not how well Hunyuan3D does (that needs real photos).
"""

from __future__ import annotations

import io
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import trimesh
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy.orm import sessionmaker

from proof3d_api.db import make_engine, make_sessionmaker
from proof3d_api.main import create_app
from proof3d_api.settings import Settings
from proof3d_api.worker import Handler, drain
from proof3d_geometry.synthetic import photo_mask, tripod_mesh
from proof3d_pipeline import StageName, WorkerKind
from proof3d_pipeline.config import GenerateConfig
from proof3d_pipeline.storage import LocalStorage
from proof3d_worker_gpu.__main__ import HANDLERS
from proof3d_worker_gpu.generate import (
    GeneratedShape,
    GeneratorUnavailableError,
    ShapeGenerator,
    generate_shape,
)
from proof3d_worker_gpu.ingest import ingest
from proof3d_worker_gpu.segment import Segmenter

CHEAP = {
    "verify": {
        "camera_fit": {
            "resolution": 64,
            "supersample": 1,
            "top_k": 2,
            "max_iter": 80,
            "polish_restarts": 0,
        }
    },
    "conditions": ["A", "B"],
}
# The product's photos; "top" is held out by ingest (last by name), so it is never sent.
VIEWS = {"a_front.png": (0.0, 10.0), "b_left.png": (90.0, 10.0), "c_back.png": (180.0, 10.0)}
LABELS = {"a_front.png": "front", "b_left.png": "left", "c_back.png": "back"}
HELD_OUT = ("z_top.png", (30.0, 60.0))


def _png(mask: np.ndarray) -> bytes:
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[..., :3] = 170
    rgba[..., 3] = mask * 255
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    return buf.getvalue()


def _glb(mesh: trimesh.Trimesh) -> bytes:
    data: bytes = mesh.export(file_type="glb")
    return data


def _alpha(png: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(png)) as img:
        assert img.mode == "RGBA"
        return np.asarray(img.getchannel("A"))


@dataclass
class FakeGenerator:
    """B gets the true product; A gets the wrong depth. Records what it was sent."""

    name: str = "fake"
    calls: list[tuple[str, dict[str, np.ndarray]]] = field(default_factory=list)
    a_glb: bytes = field(default_factory=lambda: _glb(tripod_mesh(depth_arm=0.9)))
    b_glb: bytes = field(default_factory=lambda: _glb(tripod_mesh()))

    def _shape(self, glb: bytes) -> GeneratedShape:
        return GeneratedShape(
            glb=glb,
            gpu_s=12.5,
            gpu="NVIDIA L4",
            model="tencent/x/y",
            weights_revision="rev",
            code_commit="sha",
        )

    def single(self, image: bytes, seed: int, cfg: GenerateConfig) -> GeneratedShape:
        self.calls.append(("single", {"front": _alpha(image)}))
        return self._shape(self.a_glb)

    def multi(self, views: Mapping[str, bytes], seed: int, cfg: GenerateConfig) -> GeneratedShape:
        self.calls.append(("multi", {v: _alpha(b) for v, b in views.items()}))
        return self._shape(self.b_glb)


def _no_segmenter(name: str) -> Segmenter:
    raise AssertionError("every photo is a cut-out")


def handlers(load: Any) -> dict[StageName, Handler]:
    return {
        **HANDLERS,
        StageName.INGEST: partial(ingest, load=_no_segmenter),
        StageName.GENERATE_SHAPE: partial(generate_shape, load=load),
    }


@dataclass
class Env:
    client: TestClient
    sessions: sessionmaker[Any]
    storage: LocalStorage

    def import_product(
        self, labels: dict[str, str] | None = None, config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        mesh = tripod_mesh()
        photos = {n: _png(photo_mask(mesh, az, el)) for n, (az, el) in VIEWS.items()}
        photos[HELD_OUT[0]] = _png(photo_mask(mesh, *HELD_OUT[1]))
        keys = {}
        for name, data in photos.items():
            r = self.client.post("/uploads", params={"filename": name}, content=data)
            keys[name] = r.json()["key"]
        body = {
            "photos": list(keys.values()),
            "views": {keys[n]: v for n, v in (LABELS if labels is None else labels).items()},
            "config": CHEAP if config is None else config,
        }
        r = self.client.post("/products/import", json=body)
        assert r.status_code == 201, r.text
        out: dict[str, Any] = r.json()
        return out

    def run(self, load: Any) -> int:
        return drain(self.sessions, WorkerKind.GPU, handlers(load), self.storage, 3)

    def stages(self, job_id: str) -> dict[str, dict[str, Any]]:
        return {s["stage"]: s for s in self.client.get(f"/jobs/{job_id}").json()["stages"]}


@pytest.fixture
def env(tmp_path: Path) -> Iterator[Env]:
    url = f"sqlite+pysqlite:///{tmp_path / 'db.sqlite'}"
    storage = LocalStorage(tmp_path / "store")
    settings = Settings(database_url=url, _env_file=None)  # type: ignore[call-arg]
    with TestClient(create_app(settings, create_tables=True, storage=storage)) as client:
        yield Env(client, make_sessionmaker(make_engine(url)), storage)


def test_a_and_b_are_generated_and_verified(env: Env) -> None:
    ids = env.import_product()
    fake = FakeGenerator()
    assert env.run(lambda _: fake) == 3  # ingest, generate_shape, verify

    stages = env.stages(ids["job_id"])
    assert stages["project_texture"]["status"] == "skipped"  # C is not in this job's conditions
    gen = stages["generate_shape"]
    assert gen["status"] == "succeeded", gen["error"]
    assert gen["gpu_s"] == pytest.approx(25.0)  # the generation container's time, both calls
    assert gen["summary"]["held_out_never_sent"] == ["z_top.png"]
    results = {r["condition"]: r for r in gen["summary"]["results"]}
    assert results["A"]["photos"] == {"single": "a_front.png"}
    assert results["B"]["photos"] == {
        "front": "a_front.png",
        "left": "b_left.png",
        "back": "c_back.png",
    }

    # The generator got cut-outs: transparency is the ingest mask, and no held-out photo.
    kinds = [kind for kind, _ in fake.calls]
    assert kinds == ["single", "multi"]
    sent = fake.calls[1][1]
    assert sorted(sent) == ["back", "front", "left"]
    expected = photo_mask(tripod_mesh(), 90.0, 10.0)
    assert np.array_equal(sent["left"] > 127, expected)

    compare = env.client.get(f"/products/{ids['product_id']}/compare").json()["conditions"]
    assert sorted(compare) == ["A", "B"]
    # The single-photo guess got the depth wrong; the held-out top photo catches it.
    assert compare["A"]["verdict"] == "disagrees"
    assert compare["B"]["verdict"] == "silhouette_agrees"
    assert compare["B"]["provenance"]["generator"] == "fake"
    assert compare["B"]["provenance"]["textured"] is False
    assert compare["B"]["triangles"] > 0


def test_b_needs_two_labelled_views(env: Env) -> None:
    ids = env.import_product(labels={"a_front.png": "front"})
    env.run(lambda _: FakeGenerator())
    error = env.stages(ids["job_id"])["generate_shape"]["error"]
    assert error.startswith("Condition B needs at least two input photos labelled")
    assert "leave B out of `conditions`" in error


def test_a_alone_needs_no_labels(env: Env) -> None:
    ids = env.import_product(labels={}, config={**CHEAP, "conditions": ["A"]})
    fake = FakeGenerator()
    env.run(lambda _: fake)
    gen = env.stages(ids["job_id"])["generate_shape"]
    assert gen["status"] == "succeeded", gen["error"]
    # Without labels, A takes the first input photo by name.
    assert gen["summary"]["results"][0]["photos"] == {"single": "a_front.png"}
    assert [kind for kind, _ in fake.calls] == ["single"]


def test_unavailable_generator_is_retried_then_reported(env: Env) -> None:
    ids = env.import_product()

    def unavailable(_: str) -> ShapeGenerator:
        raise GeneratorUnavailableError("the Modal app is not reachable")

    env.run(unavailable)
    gen = env.stages(ids["job_id"])["generate_shape"]
    assert gen["status"] == "failed"
    assert gen["attempts"] == 3
    assert gen["error"].startswith("Shape generation is unavailable: the Modal app")


class CrashingGenerator(FakeGenerator):
    def multi(self, views: Mapping[str, bytes], seed: int, cfg: GenerateConfig) -> GeneratedShape:
        raise RuntimeError("CUDA out of memory")


def test_a_failed_generation_call_is_retried_and_stores_nothing(env: Env) -> None:
    ids = env.import_product()
    env.run(lambda _: CrashingGenerator())
    gen = env.stages(ids["job_id"])["generate_shape"]
    assert gen["status"] == "failed"
    assert gen["error"] == (
        "The generation container failed (RuntimeError: CUDA out of memory). Retrying."
    )
    compare = env.client.get(f"/products/{ids['product_id']}/compare").json()
    assert compare["conditions"] == {}  # A succeeded, but nothing is kept from a failed stage


def test_a_model_that_is_not_glb_is_refused(env: Env) -> None:
    ids = env.import_product(config={**CHEAP, "conditions": ["A"]})
    env.run(lambda _: FakeGenerator(a_glb=b"glTF" + b"\0" * 40))
    error = env.stages(ids["job_id"])["generate_shape"]["error"]
    assert error == "The generator's model for condition A is not a GLB."


def test_import_checks_views_and_conditions(env: Env) -> None:
    r = env.client.post(
        "/products/import",
        json={"photos": ["s3://b/front.png"], "views": {"s3://b/other.png": "front"}},
    )
    assert r.status_code == 422
    assert "views must label photos listed in photos" in r.text
    r = env.client.post(
        "/products/import", json={"photos": ["s3://b/front.png"], "config": {"conditions": ["C"]}}
    )
    assert r.status_code == 422
    assert "condition C needs condition B" in r.text
    r = env.client.post(
        "/products/import",
        json={"photos": ["s3://b/front.png"], "config": {"conditions": ["A", "B", "C"]}},
    )
    stages = env.stages(r.json()["job_id"])
    assert stages["project_texture"]["status"] == "pending"
    assert stages["repair"]["status"] == "skipped"
