from __future__ import annotations

import pathlib
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from proof3d_api import queue
from proof3d_api.db import AssetVersion, Base, Job, Photo, Score, make_engine, make_sessionmaker
from proof3d_api.main import create_app
from proof3d_api.settings import Settings
from proof3d_pipeline import StageMetrics, StageName, StageResult, StageStatus, WorkerKind

DB_URL = "sqlite+pysqlite:///:memory:"


@pytest.fixture
def settings() -> Settings:
    return Settings(database_url=DB_URL)


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings, create_tables=True)) as c:
        yield c


@pytest.fixture
def session() -> Iterator[Session]:
    engine = make_engine(DB_URL)
    Base.metadata.create_all(engine)
    with make_sessionmaker(engine)() as s:
        yield s


def _ok(stage: StageName) -> StageResult:
    return StageResult(
        stage=stage, status=StageStatus.SUCCEEDED, metrics=StageMetrics(wall_s=1.5, gpu_s=1.5)
    )


def test_import_creates_queued_job(client: TestClient) -> None:
    r = client.post("/products/import", json={"photos": ["s3://demo/front.jpg"], "title": "Mug"})
    assert r.status_code == 201
    job = client.get(f"/jobs/{r.json()['job_id']}").json()
    assert job["stages"][0]["stage"] == "ingest"
    assert job["stages"][0]["status"] == "queued"
    assert all(s["status"] == "pending" for s in job["stages"][1:])
    assert job["total_cost_usd"] is None  # nothing measured yet, so nothing invented


def test_import_requires_input(client: TestClient) -> None:
    assert client.post("/products/import", json={}).status_code == 422


def test_import_rejects_bad_config(client: TestClient) -> None:
    r = client.post("/products/import", json={"photos": ["a.jpg"], "config": {"bogus": 1}})
    assert r.status_code == 422


def test_unknown_ids_404(client: TestClient) -> None:
    assert client.get("/jobs/nope").status_code == 404
    assert client.get("/assets/nope/scorecard").status_code == 404
    assert client.get("/products/nope/compare").status_code == 404


def _new_job(session: Session) -> Job:
    from proof3d_api.db import Product

    product = Product(title="Mug")
    job = Job(product=product, pipeline_version="t", config={}, config_hash="h")
    queue.create_stage_runs(job)
    session.add_all([product, job])
    session.flush()
    return job


def test_queue_runs_stages_in_order(session: Session) -> None:
    job = _new_job(session)
    assert queue.claim(session, WorkerKind.CPU) is None  # ingest is a GPU stage
    seen = []
    for _ in range(8):
        run = None
        for kind in WorkerKind:
            run = queue.claim(session, kind)
            if run:
                break
        assert run is not None
        seen.append(run.stage)
        queue.complete(session, run, _ok(StageName(run.stage)))
    assert seen == [s.value for s in StageName]
    assert all(s.status == "succeeded" for s in job.stages)


def test_queue_retries_then_fails(session: Session) -> None:
    job = _new_job(session)
    for attempt in range(1, 4):
        run = queue.claim(session, WorkerKind.GPU)
        assert run is not None
        assert run.attempts == attempt
        queue.fail(session, run, "No usable photos found", max_attempts=3)
    assert job.stages[0].status == "failed"
    assert job.stages[0].error == "No usable photos found"
    assert queue.claim(session, WorkerKind.GPU) is None


def test_rerun_requires_earlier_stages(session: Session) -> None:
    job = _new_job(session)
    with pytest.raises(queue.RerunConflictError, match="earlier stage"):
        queue.rerun_from(session, job, StageName.VERIFY)


def test_rerun_resets_later_stages(session: Session) -> None:
    job = _new_job(session)
    for _ in range(4):  # ingest .. verify
        run = queue.claim(session, WorkerKind.GPU)
        assert run is not None
        queue.complete(session, run, _ok(StageName(run.stage)))
    queue.rerun_from(session, job, StageName.PROJECT_TEXTURE)
    status = {s.stage: s.status for s in job.stages}
    assert status["generate_shape"] == "succeeded"
    assert status["project_texture"] == "queued"
    assert status["verify"] == "pending"


def test_compare_returns_latest_version_per_condition(tmp_path: pathlib.Path) -> None:
    url = f"sqlite+pysqlite:///{tmp_path / 'db.sqlite'}"
    with TestClient(create_app(Settings(database_url=url), create_tables=True)) as client:
        r = client.post(
            "/products/import", json={"photos": ["s3://demo/a.jpg", "s3://demo/b.jpg"]}
        ).json()
        with make_sessionmaker(make_engine(url))() as s, s.begin():
            for method, glb in [("A", "a1.glb"), ("C", "c1.glb"), ("C", "c2.glb")]:
                s.add(AssetVersion(job_id=r["job_id"], method=method, glb_key=glb))
                s.flush()
        body = client.get(f"/products/{r['product_id']}/compare").json()
        assert sorted(body["conditions"]) == ["A", "C"]
        c_id = body["conditions"]["C"]["asset_id"]
        card = client.get(f"/assets/{c_id}/scorecard").json()
        assert card["method"] == "C"
        assert card["scores"] == []


def test_scorecard_reports_stored_scores(session: Session) -> None:
    job = _new_job(session)
    photo = Photo(product_id=job.product_id, source="held.jpg", role="held_out")
    asset = AssetVersion(job_id=job.id, method="C", glb_key="assets/x.glb")
    session.add_all([photo, asset])
    session.flush()
    session.add(
        Score(
            asset_id=asset.id,
            photo_id=photo.id,
            fit_iou=0.93,
            fit_ambiguous=False,
            metrics={"silhouette_iou": 0.93, "lpips_masked": 0.12},
            passed=True,
        )
    )
    session.flush()
    from proof3d_api.main import _scorecard

    card = _scorecard(asset)
    assert card.method == "C"
    assert card.scores[0].metrics["lpips_masked"] == 0.12
