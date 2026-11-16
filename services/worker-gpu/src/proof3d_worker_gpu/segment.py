"""Background removal: a product mask for a photo that has no transparency (PRD 5.1).

BiRefNet runs through ONNX Runtime from the authors' own ONNX export, so no remote model code
is executed and the same file runs on a laptop CPU and, with the CUDA provider, on a GPU. The
weights are downloaded once into ``PROOF3D_MODEL_DIR`` and checked against a pinned SHA-256
before they are loaded. Licence and training-data terms: docs/licence-register.md.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from proof3d_api.settings import get_settings

log = logging.getLogger(__name__)

FloatArray = NDArray[np.float32]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class SegmenterModel:
    name: str
    url: str
    sha256: str
    input_size: int  # the ONNX export takes a fixed square input

    @property
    def filename(self) -> str:
        return self.url.rsplit("/", 1)[1]


MODELS = {
    m.name: m
    for m in [
        # Swin-v1-tiny backbone, "general" training mix, 214 MB. On an 8-core laptop CPU: ~17 s
        # to load, 13-19 s and a peak of ~6 GB RAM per photo.
        SegmenterModel(
            name="birefnet-general-lite",
            url="https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/"
            "BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx",
            sha256="5600024376f572a557870a5eb0afb1e5961636bef4e1e22132025467d0f03333",
            input_size=1024,
        ),
    ]
}


class Segmenter(Protocol):
    name: str

    def __call__(self, image: Image.Image) -> FloatArray:
        """Product probability per pixel, shaped like the image (height, width)."""
        ...


class ModelUnavailableError(RuntimeError):
    pass


class SegmenterOutOfMemoryError(MemoryError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_model(model: SegmenterModel, model_dir: Path) -> Path:
    """The verified local copy of ``model``, downloading it first if needed."""
    path = model_dir / model.filename
    if path.exists():
        if _sha256(path) == model.sha256:
            return path
        log.warning("model file has the wrong hash; downloading again", extra={"path": str(path)})
    model_dir.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    log.info("downloading model", extra={"model": model.name, "url": model.url})
    try:
        with urllib.request.urlopen(model.url, timeout=60) as resp, part.open("wb") as out:
            shutil.copyfileobj(resp, out, 2**20)
    except OSError as exc:
        part.unlink(missing_ok=True)
        raise ModelUnavailableError(f"could not download {model.url}: {exc}") from exc
    got = _sha256(part)
    if got != model.sha256:
        part.unlink()
        raise ModelUnavailableError(
            f"{model.url} has SHA-256 {got}, expected {model.sha256}; refusing to load it"
        )
    part.replace(path)
    return path


class OnnxSegmenter:
    """BiRefNet through ONNX Runtime. Uses CUDA when onnxruntime-gpu provides it."""

    def __init__(self, model: SegmenterModel, path: Path, session: Any = None) -> None:
        self.name = model.name
        self.size = model.input_size
        if session is None:  # pragma: no cover - needs the real weights; see test_segment.py
            import onnxruntime as ort

            wanted = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            providers = [p for p in wanted if p in ort.get_available_providers()]
            options = ort.SessionOptions()
            # Without the arena and pre-planned buffers, the CPU peak drops from ~7.1 to ~5.9 GB
            # per 1024 px photo, for ~1.5 s more per photo.
            options.enable_cpu_mem_arena = False
            options.enable_mem_pattern = False
            session = ort.InferenceSession(str(path), options, providers=providers)
        self.session = session
        self.input_name: str = session.get_inputs()[0].name

    def __call__(self, image: Image.Image) -> FloatArray:
        rgb = image.convert("RGB")
        resized = rgb.resize((self.size, self.size), Image.Resampling.BILINEAR)
        x = (np.asarray(resized, dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        try:
            logits = self.session.run(None, {self.input_name: x.transpose(2, 0, 1)[None]})[0]
        except Exception as exc:  # ONNX Runtime reports allocation failures as its own Fail
            if isinstance(exc, MemoryError) or "bad allocation" in str(exc):
                raise SegmenterOutOfMemoryError(str(exc)) from exc
            raise
        prob = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float32)[0, 0]))
        back = Image.fromarray(prob.astype(np.float32), mode="F").resize(
            rgb.size, Image.Resampling.BILINEAR
        )
        return np.clip(np.asarray(back, dtype=np.float32), 0.0, 1.0)


@lru_cache(maxsize=2)
def load_segmenter(name: str) -> Segmenter:  # pragma: no cover - needs the real weights
    """The named segmenter, loaded once per process (loading takes ~20 s on a laptop CPU)."""
    model = MODELS[name]
    return OnnxSegmenter(model, ensure_model(model, Path(get_settings().model_dir)))
