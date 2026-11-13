from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from proof3d_pipeline import (
    STAGE_ORDER,
    PipelineConfig,
    StageName,
    WorkerKind,
    config_hash,
    load_config,
    stages_from,
    worker_for,
)
from proof3d_pipeline.measure import measure
from proof3d_pipeline.storage import LocalStorage, content_key

REPO = Path(__file__).resolve().parents[3]


def test_stage_order_matches_prd() -> None:
    assert [s.value for s in STAGE_ORDER] == [
        "ingest",
        "generate_shape",
        "project_texture",
        "verify",
        "repair",
        "optimize",
        "phone_test",
        "publish",
    ]


def test_rerun_from_stage() -> None:
    assert stages_from(StageName.VERIFY)[0] is StageName.VERIFY
    assert stages_from(StageName.PUBLISH) == (StageName.PUBLISH,)
    assert stages_from(StageName.INGEST) == STAGE_ORDER


def test_every_stage_has_a_worker() -> None:
    assert {worker_for(s) for s in STAGE_ORDER} == set(WorkerKind)


def test_config_hash_stable_and_sensitive() -> None:
    a, b = PipelineConfig(), PipelineConfig()
    assert config_hash(a) == config_hash(b)
    c = PipelineConfig.model_validate({"verify": {"camera_fit": {"resolution": 256}}})
    assert config_hash(c) != config_hash(a)


def test_config_rejects_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        PipelineConfig.model_validate({"verify": {"camera_fitt": {}}})


def test_shipped_eval_configs_validate() -> None:
    configs = sorted((REPO / "eval" / "configs").glob("*.yaml"))
    assert configs, "no eval configs found"
    for path in configs:
        load_config(path)


def test_local_storage_roundtrip(tmp_path: Path) -> None:
    store = LocalStorage(tmp_path)
    key = content_key("renders", b"abc", ".png")
    assert not store.exists(key)
    store.put(key, b"abc")
    assert store.get(key) == b"abc"
    assert key.endswith(".png")


def test_local_storage_rejects_escape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="escapes"):
        LocalStorage(tmp_path).put("../outside", b"x")


def test_measure_records_time_and_gpu() -> None:
    with measure(on_gpu=True) as m:
        sum(range(10_000))
    metrics = m.metrics(cost_usd=0.01)
    assert metrics.wall_s > 0
    assert metrics.gpu_s == metrics.wall_s
    assert metrics.cost_usd == 0.01


def test_measure_records_peak_memory_held_at_the_end() -> None:
    with measure() as baseline:
        pass
    with measure() as m:
        blob = b"" * (64 * 2**20)  # written, so the pages are resident
    assert len(blob) == 64 * 2**20
    assert m.peak_mem_mb >= baseline.peak_mem_mb + 48
    assert m.gpu_s == 0.0
