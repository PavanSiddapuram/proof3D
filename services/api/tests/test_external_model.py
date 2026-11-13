"""Bring-your-own-model import: verify a model another generator made, without regenerating."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from proof3d_api import queue
from proof3d_api.db import Base, Job, Product, make_engine, make_sessionmaker
from proof3d_api.main import create_app
from proof3d_api.settings import Settings
from proof3d_pipeline import StageMetrics, StageName, StageResult, StageStatus, WorkerKind

DB_URL = "sqlite+pysqlite:///:memory:"
PHOTOS = ["s3://demo/front.jpg", "s3://demo/left.jpg", "s3://demo/back.jpg"]
SKIPPED = {StageName.GENERATE_SHAPE, StageName.PROJECT_TEXTURE}


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(create_app(Settings(database_url=DB_URL), create_tables=True)) as c:
        yield c


@pytest.fixture
def session() -> Iterator[Session]:
    engine = make_engine(DB_URL)
    Base.metadata.create_all(engine)
    with make_sessionmaker(engine)() as s:
        yield s


def _request(**model: Any) -> dict[str, Any]:
    body = {"glb": "s3://demo/versa.glb", "generator": "versa_ai", "held_out": [PHOTOS[2]]}
    body.update(model)
    return {"photos": PHOTOS, "title": "Sneaker", "model": body}


def _ok(stage: StageName) -> StageResult:
    return StageResult(stage=stage, status=StageStatus.SUCCEEDED, metrics=StageMetrics(wall_s=1))


def _external_job(session: Session) -> Job:
    job = Job(product=Product(title="Sneaker"), pipeline_version="t", config={}, config_hash="h")
    queue.create_stage_runs(job, skip=SKIPPED)
    session.add(job)
    session.flush()
    return job


def test_import_registers_the_model_and_skips_generation(client: TestClient) -> None:
    r = client.post("/products/import", json=_request(generator_version="2026-10"))
    assert r.status_code == 201
    body = r.json()
    assert body["asset_id"]

    stages = {
        s["stage"]: s["status"] for s in client.get(f"/jobs/{body['job_id']}").json()["stages"]
    }
    assert stages["ingest"] == "queued"
    assert stages["generate_shape"] == "skipped"
    assert stages["project_texture"] == "skipped"
    assert stages["verify"] == "pending"

    card = client.get(f"/assets/{body['asset_id']}/scorecard").json()
    assert card["method"] == "external"
    assert card["provenance"] == {
        "generator": "versa_ai",
        "generator_version": "2026-10",
        "imported": True,
    }
    compare = client.get(f"/products/{body['product_id']}/compare").json()
    assert list(compare["conditions"]) == ["external"]


def test_import_records_which_photos_the_generator_saw(tmp_path: Path) -> None:
    url = f"sqlite+pysqlite:///{tmp_path / 'db.sqlite'}"
    with TestClient(create_app(Settings(database_url=url), create_tables=True)) as c:
        body = c.post("/products/import", json=_request()).json()
    with make_sessionmaker(make_engine(url))() as s:
        product = s.get(Product, body["product_id"])
        assert product is not None
        roles = {p.source: p.role for p in product.photos}
    assert roles == {PHOTOS[0]: "input", PHOTOS[1]: "input", PHOTOS[2]: "held_out"}


def test_plain_import_leaves_roles_to_ingest(client: TestClient) -> None:
    body = client.post("/products/import", json={"photos": PHOTOS}).json()
    assert body["asset_id"] is None
    stages = client.get(f"/jobs/{body['job_id']}").json()["stages"]
    assert all(s["status"] != "skipped" for s in stages)


@pytest.mark.parametrize(
    "model",
    [
        {"held_out": []},  # nothing held out: nothing could verify the model
        {"held_out": ["s3://demo/not-listed.jpg"]},
        {"generator": ""},
        {"glb": ""},
        {"unexpected": 1},
    ],
)
def test_import_rejects_bad_external_models(client: TestClient, model: dict[str, Any]) -> None:
    assert client.post("/products/import", json=_request(**model)).status_code == 422


def test_queue_jumps_over_skipped_stages(session: Session) -> None:
    job = _external_job(session)
    ran = []
    while (run := queue.claim(session, WorkerKind.GPU)) is not None:
        ran.append(run.stage)
        queue.complete(session, run, _ok(StageName(run.stage)))
    assert ran == ["ingest", "verify", "repair"]
    status = {s.stage: s.status for s in job.stages}
    assert status["generate_shape"] == "skipped"
    assert status["optimize"] == "queued"


def test_rerun_keeps_planned_skips(session: Session) -> None:
    job = _external_job(session)
    for _ in range(2):  # ingest, verify
        run = queue.claim(session, WorkerKind.GPU)
        assert run is not None
        queue.complete(session, run, _ok(StageName(run.stage)))
    with pytest.raises(queue.RerunConflictError, match="not part of this job"):
        queue.rerun_from(session, job, StageName.GENERATE_SHAPE)

    queue.rerun_from(session, job, StageName.VERIFY)  # earlier skipped stages count as done
    status = {s.stage: s.status for s in job.stages}
    assert status["verify"] == "queued"
    assert status["project_texture"] == "skipped"


def test_refused_rerun_changes_nothing(session: Session) -> None:
    job = Job(product=Product(title="Mug"), pipeline_version="t", config={}, config_hash="h")
    queue.create_stage_runs(job)
    session.add(job)
    session.flush()
    publish = next(s for s in job.stages if s.stage == StageName.PUBLISH.value)
    publish.status = StageStatus.SUCCEEDED.value  # e.g. from an earlier successful run
    before = {s.stage: s.status for s in job.stages}
    with pytest.raises(queue.RerunConflictError, match="earlier stage"):
        queue.rerun_from(session, job, StageName.VERIFY)  # ingest has not succeeded
    assert {s.stage: s.status for s in job.stages} == before
