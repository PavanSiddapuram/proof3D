"""Background removal: model download and checks, pre- and post-processing, and (opt-in) the
real BiRefNet weights on synthetic photos."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from proof3d_geometry.synthetic import opaque_photo, tripod_mesh
from proof3d_worker_gpu.segment import (
    MODELS,
    ModelUnavailableError,
    OnnxSegmenter,
    SegmenterModel,
    SegmenterOutOfMemoryError,
    ensure_model,
)


def _model(source: Path, sha256: str | None = None) -> SegmenterModel:
    digest = sha256 or hashlib.sha256(source.read_bytes()).hexdigest()
    return SegmenterModel(name="t", url=source.as_uri(), sha256=digest, input_size=8)


def test_ensure_model_downloads_verifies_and_reuses(tmp_path: Path) -> None:
    source = tmp_path / "src" / "weights.onnx"
    source.parent.mkdir()
    source.write_bytes(b"weights v1")
    model = _model(source)
    cache = tmp_path / "models"

    path = ensure_model(model, cache)
    assert path == cache / "weights.onnx"
    assert path.read_bytes() == b"weights v1"

    source.write_bytes(b"changed upstream")  # a good cached copy is not downloaded again
    assert ensure_model(model, cache).read_bytes() == b"weights v1"

    path.write_bytes(b"corrupt")  # a bad cached copy is replaced, and the new one checked
    with pytest.raises(ModelUnavailableError, match="refusing to load it"):
        ensure_model(model, cache)
    assert list(cache.iterdir()) == [path]  # no half-written .part file is left behind


def test_ensure_model_reports_download_failures(tmp_path: Path) -> None:
    missing = _model(tmp_path / "nowhere.onnx", sha256="0" * 64)
    with pytest.raises(ModelUnavailableError, match="could not download"):
        ensure_model(missing, tmp_path / "models")


class FakeSession:
    """Stands in for onnxruntime.InferenceSession: logits +8 in the left half, -8 elsewhere."""

    def __init__(self) -> None:
        self.inputs: list[np.ndarray] = []

    def get_inputs(self) -> list[Any]:
        return [type("Input", (), {"name": "input_image"})()]

    def run(self, _: object, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        x = feed["input_image"]
        self.inputs.append(x)
        logits = np.full((1, 1, *x.shape[2:]), -8.0, dtype=np.float32)
        logits[..., : x.shape[3] // 2] = 8.0
        return [logits]


def test_onnx_segmenter_pre_and_post_processing() -> None:
    session = FakeSession()
    seg = OnnxSegmenter(MODELS["birefnet-general-lite"], Path("unused"), session=session)
    white = Image.new("RGB", (40, 30), (255, 255, 255))

    prob = seg(white)
    x = session.inputs[0]
    assert x.shape == (1, 3, 1024, 1024)
    assert x.dtype == np.float32
    # ImageNet normalisation of pure white, per channel.
    assert x[0, :, 0, 0] == pytest.approx([2.2489, 2.4286, 2.64], abs=1e-3)
    assert prob.shape == (30, 40)  # back at the photo's own size
    assert prob[:, :18].min() > 0.99
    assert prob[:, 22:].max() < 0.01


class FailingSession(FakeSession):
    def __init__(self, message: str) -> None:
        super().__init__()
        self.message = message

    def run(self, _: object, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        raise RuntimeError(self.message)


def test_onnx_segmenter_reports_running_out_of_memory() -> None:
    model = MODELS["birefnet-general-lite"]
    image = Image.new("RGB", (8, 8))
    oom = OnnxSegmenter(model, Path("unused"), session=FailingSession("FAIL : bad allocation"))
    with pytest.raises(SegmenterOutOfMemoryError):
        oom(image)
    other = OnnxSegmenter(model, Path("unused"), session=FailingSession("shape mismatch"))
    with pytest.raises(RuntimeError, match="shape mismatch"):
        other(image)


@pytest.mark.skipif(
    not os.environ.get("PROOF3D_RUN_MODEL_TESTS"),
    reason="needs the real BiRefNet weights (214 MB), ~6 GB of free RAM and ~1 min of CPU; "
    "set PROOF3D_RUN_MODEL_TESTS=1 to run",
)
def test_real_birefnet_finds_the_synthetic_product() -> None:  # pragma: no cover
    from proof3d_worker_gpu.segment import load_segmenter

    seg = load_segmenter("birefnet-general-lite")
    for az, el in [(0.0, 10.0), (30.0, 60.0)]:
        rgb, truth = opaque_photo(tripod_mesh(), az, el)
        mask = seg(Image.fromarray(rgb)) >= 0.5
        assert (mask & truth).sum() / (mask | truth).sum() > 0.95
