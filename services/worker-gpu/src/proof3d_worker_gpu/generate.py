"""Generate shape (PRD 5.2): the meshes for conditions A and B.

* **A, single photo:** Hunyuan3D-2 on one input photo, the front one when it is labelled.
* **B, multiple photos:** Hunyuan3D-2mv on the input photos labelled front, left, back and
  right, one per side (the model's four view slots).

Both see the same cut-out photos (the photo with its ingest mask as transparency) and the same
seed, so the difference between A and B is the extra views (ADR 0005). The models run in their
own Modal container (``services/gen-hunyuan3d``, ADR 0009); this stage sends them photos and
stores the meshes they return as asset versions. Held-out photos are never sent.

Shape only for now: the generated texture for A and B (Hunyuan3D-Paint) is not built yet.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol

import numpy as np
import trimesh
from PIL import Image

from proof3d_api.db import AssetVersion, Photo, Product
from proof3d_api.settings import get_settings
from proof3d_api.worker import StageContext, StageError
from proof3d_pipeline import Condition, PhotoRole, StageName, StageResult, StageStatus, ViewLabel
from proof3d_pipeline.config import GenerateConfig
from proof3d_pipeline.measure import measure
from proof3d_pipeline.storage import Storage, content_key
from proof3d_worker_gpu.ingest import decode_mask, photo_name

log = logging.getLogger(__name__)

# Hunyuan3D-2mv's view slots, in its own order (hy3dgen MVImageProcessorV2.view2idx).
MULTI_VIEWS = (ViewLabel.FRONT, ViewLabel.LEFT, ViewLabel.BACK, ViewLabel.RIGHT)
GEN_APP = "proof3d-gen-hunyuan3d"


@dataclass(frozen=True)
class GeneratedShape:
    glb: bytes
    gpu_s: float  # GPU time of this call in the generation container
    gpu: str
    model: str  # "repo/subfolder" of the weights
    weights_revision: str
    code_commit: str


class ShapeGenerator(Protocol):
    name: str

    def single(self, image: bytes, seed: int, cfg: GenerateConfig) -> GeneratedShape: ...

    def multi(
        self, views: Mapping[str, bytes], seed: int, cfg: GenerateConfig
    ) -> GeneratedShape: ...


class GeneratorUnavailableError(RuntimeError):
    pass


class ModalHunyuan3D:  # pragma: no cover - calls the deployed Modal app; see docs/gpu.md
    """Calls the ``Hunyuan3D`` class deployed by ``make gen-deploy``."""

    name = "hunyuan3d-2"

    def __init__(self) -> None:
        import modal

        try:
            self._remote = modal.Cls.from_name(GEN_APP, "Hunyuan3D")()
        except Exception as exc:
            raise GeneratorUnavailableError(
                f"the Modal app {GEN_APP} is not reachable ({exc}); deploy it with "
                "`make gen-deploy` and check `modal token` is set"
            ) from exc

    def single(self, image: bytes, seed: int, cfg: GenerateConfig) -> GeneratedShape:
        return GeneratedShape(**self._remote.single.remote(image, seed, cfg.model_dump()))

    def multi(self, views: Mapping[str, bytes], seed: int, cfg: GenerateConfig) -> GeneratedShape:
        return GeneratedShape(**self._remote.multi.remote(dict(views), seed, cfg.model_dump()))


@lru_cache(maxsize=1)
def load_generator(backend: str) -> ShapeGenerator:  # pragma: no cover - needs Modal
    if backend != "hunyuan3d-2":
        raise GeneratorUnavailableError(f"unknown generation backend {backend!r}")
    return ModalHunyuan3D()


def cutout(storage: Storage, photo: Photo) -> bytes:
    """The photo as an RGBA PNG whose transparency is the ingest mask."""
    assert photo.mask_key is not None
    with Image.open(io.BytesIO(storage.get(photo.source))) as img:
        rgb = np.asarray(img.convert("RGB"))
    mask = decode_mask(storage.get(photo.mask_key))
    if mask.shape != rgb.shape[:2]:
        raise StageError(f"The mask of {photo_name(photo)} does not match it; rerun from ingest.")
    rgba = np.dstack([rgb, mask.astype(np.uint8) * 255])
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    return buf.getvalue()


def pick_single(inputs: list[Photo]) -> Photo:
    """A's photo: the front one if labelled, else the first input by name."""
    front = [p for p in inputs if p.view_label == ViewLabel.FRONT.value]
    return (front or inputs)[0]


def pick_multi(inputs: list[Photo], limit: int | None) -> dict[ViewLabel, Photo]:
    """B's photos: the first input by name for each of the model's view slots that has one."""
    chosen: dict[ViewLabel, Photo] = {}
    for view in MULTI_VIEWS:
        labelled = [p for p in inputs if p.view_label == view.value]
        if labelled and (limit is None or len(chosen) < limit):
            chosen[view] = labelled[0]
    return chosen


def _asset(
    ctx: StageContext,
    condition: Condition,
    shape: GeneratedShape,
    generator: str,
    photos: dict[str, str],
) -> AssetVersion:
    try:
        mesh = trimesh.load(io.BytesIO(shape.glb), file_type="glb", force="mesh")
    except Exception as exc:
        raise StageError(f"The generator's model for condition {condition} is not a GLB.") from exc
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise StageError(f"The generator returned an empty model for condition {condition}.")
    key = ctx.storage.put(content_key("meshes", shape.glb, ".glb"), shape.glb)
    cfg = ctx.config
    asset = AssetVersion(
        job_id=ctx.job_id,
        method=condition.value,
        glb_key=key,
        triangles=len(mesh.faces),
        bytes=len(shape.glb),
        provenance={
            "generator": generator,
            "model": shape.model,
            "weights_revision": shape.weights_revision,
            "code_commit": shape.code_commit,
            "gpu": shape.gpu,
            "seed": cfg.seed,
            "generate": cfg.generate.model_dump(mode="json"),
            "photos": photos,  # view slot -> photo name
            "textured": False,
        },
    )
    ctx.session.add(asset)
    ctx.session.flush()
    return asset


def generate_shape(
    ctx: StageContext, load: Callable[[str], ShapeGenerator] = load_generator
) -> StageResult:
    cfg = ctx.config
    wanted = set(cfg.conditions)
    with measure() as meter:  # the GPU time is the generation container's, added below
        product = ctx.session.get(Product, ctx.product_id)
        assert product is not None, "a job always has a product"
        inputs = sorted(
            (p for p in product.photos if p.role == PhotoRole.INPUT.value and p.mask_key),
            key=lambda p: (photo_name(p), p.source),
        )
        if not inputs:
            raise StageError("No input photos with masks to generate from. Rerun from ingest.")

        multi: dict[ViewLabel, Photo] = {}
        if Condition.B_MULTI_GENERATED in wanted:
            multi = pick_multi(inputs, cfg.n_input_photos)
            if len(multi) < 2:
                labels = {photo_name(p): p.view_label for p in inputs}
                raise StageError(
                    "Condition B needs at least two input photos labelled front, left, back or "
                    f"right; the input photos are labelled {labels}. Label them with `views` "
                    "on import, or leave B out of `conditions`."
                )

        try:
            generator = load(cfg.generate.backend)
        except GeneratorUnavailableError as exc:
            raise StageError(f"Shape generation is unavailable: {exc}.", retryable=True) from exc

        shapes: list[tuple[Condition, GeneratedShape, dict[str, str]]] = []
        try:
            if Condition.A_SINGLE_GENERATED in wanted:
                photo = pick_single(inputs)
                shape = generator.single(cutout(ctx.storage, photo), cfg.seed, cfg.generate)
                shapes.append((Condition.A_SINGLE_GENERATED, shape, {"single": photo_name(photo)}))
            if multi:
                views = {v.value: cutout(ctx.storage, p) for v, p in multi.items()}
                shape = generator.multi(views, cfg.seed, cfg.generate)
                used = {v.value: photo_name(p) for v, p in multi.items()}
                shapes.append((Condition.B_MULTI_GENERATED, shape, used))
        except (StageError, GeneratorUnavailableError):
            raise
        except Exception as exc:  # remote failures: out of GPU memory, preemption, timeouts
            log.exception("generation call failed", extra={"job_id": ctx.job_id})
            raise StageError(
                f"The generation container failed ({type(exc).__name__}: {exc}). Retrying.",
                retryable=True,
            ) from exc

        outputs: dict[str, str] = {}
        results: list[dict[str, Any]] = []
        for condition, shape, used in shapes:
            asset = _asset(ctx, condition, shape, generator.name, used)
            outputs[f"asset/{condition.value}"] = asset.id
            results.append(
                {
                    "condition": condition.value,
                    "asset_id": asset.id,
                    "photos": used,
                    "triangles": asset.triangles,
                    "gpu_s": round(shape.gpu_s, 2),
                    "gpu": shape.gpu,
                }
            )

    gpu_s = sum(shape.gpu_s for _, shape, _ in shapes)
    price = get_settings().gen_gpu_usd_per_hour
    metrics = meter.metrics(cost_usd=gpu_s * price / 3600 if price > 0 else None)
    return StageResult(
        stage=StageName.GENERATE_SHAPE,
        status=StageStatus.SUCCEEDED,
        metrics=metrics.model_copy(update={"gpu_s": gpu_s}),
        outputs=outputs,
        summary={
            "results": results,
            "held_out_never_sent": sorted(
                photo_name(p) for p in product.photos if p.role == PhotoRole.HELD_OUT.value
            ),
            "textured": False,
            "not_built_yet": "generated texture for A and B (Hunyuan3D-Paint)",
        },
    )
