"""Ingest (PRD 5.1), the part that runs without models.

For each photo: read it from storage, take the product mask from its transparency, flag
problems, store the mask, and settle its role. What is not built yet, and what stands in:

* background removal (BiRefNet): photos must arrive background-removed, as PNGs whose
  transparency marks the product. An opaque photo is rejected with a message saying so;
* view labels (orientation model or VLM): photos get none, so camera fitting searches the
  whole orbit;
* fetching photos from a URL: photos must be uploaded (``POST /uploads``) first.

Roles given at import (an imported model's held-out photos) are kept. Otherwise one usable
photo is held out. Without view labels it cannot prefer an angle missing from the inputs
(PRD 5.1), so it takes the last photo by name, which is at least repeatable.
"""

from __future__ import annotations

import io
from pathlib import PurePosixPath

import numpy as np
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError
from scipy import ndimage

from proof3d_api.db import Photo, Product
from proof3d_api.worker import StageContext, StageError
from proof3d_pipeline import PhotoRole, StageName, StageResult, StageStatus
from proof3d_pipeline.measure import measure
from proof3d_pipeline.storage import Storage, content_key

BoolArray = NDArray[np.bool_]

MIN_PRODUCT_SHARE = 0.005  # below this share of the frame the mask is treated as empty
SECOND_OBJECT_SHARE = 0.10  # a second blob this large relative to the first is flagged

# Flags that make a photo unusable, with the reason shown to the merchant.
REJECTING = {
    "unreadable": "the file could not be read as an image",
    "needs_background_removal": (
        "it has no transparent background; automatic background removal is not built yet, "
        "so upload a background-removed PNG"
    ),
    "empty_mask": "no product was found in it",
}
# Flags that are reported but keep the photo in use.
WARNINGS = {
    "touches_border": "the product touches the edge of the frame and may be cropped",
    "multiple_objects": "it shows more than one separate object",
}


def photo_name(photo: Photo) -> str:
    return PurePosixPath(photo.source).name or photo.source


def mask_from_image(data: bytes) -> tuple[BoolArray | None, list[str]]:
    """The product mask from a background-removed image's transparency, and quality flags."""
    try:
        with Image.open(io.BytesIO(data)) as opened:
            opened.load()
            img: Image.Image = opened
            if img.mode == "P" and "transparency" in img.info:
                img = img.convert("RGBA")
            if "A" not in img.getbands():
                return None, ["needs_background_removal"]
            alpha = np.asarray(img.getchannel("A"))
    except (UnidentifiedImageError, OSError):
        return None, ["unreadable"]
    mask = alpha > 127
    if mask.all():  # an alpha channel that hides nothing is not a cut-out
        return None, ["needs_background_removal"]
    if mask.mean() < MIN_PRODUCT_SHARE:
        return None, ["empty_mask"]

    flags = []
    if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
        flags.append("touches_border")
    labels, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    if n > 1:
        sizes = np.sort(np.bincount(labels.ravel())[1:])[::-1]
        if sizes[1] >= SECOND_OBJECT_SHARE * sizes[0]:
            flags.append("multiple_objects")
    return mask, flags


def encode_mask(mask: BoolArray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def decode_mask(data: bytes) -> BoolArray:
    with Image.open(io.BytesIO(data)) as img:
        return np.asarray(img.convert("L")) > 127


def _read(storage: Storage, photo: Photo) -> bytes:
    name = photo_name(photo)
    if "://" in photo.source:
        raise StageError(
            f"Photo {name} is a URL, and fetching photos from URLs is not built yet. Upload it "
            "with POST /uploads and import the returned key."
        )
    if not storage.exists(photo.source):
        raise StageError(f"Photo {name} is not in storage; upload it again.")
    return storage.get(photo.source)


def ingest(ctx: StageContext) -> StageResult:
    with measure() as meter:  # CPU work, even when the GPU worker runs it
        product = ctx.session.get(Product, ctx.product_id)
        assert product is not None, "a job always has a product"
        photos = sorted(product.photos, key=lambda p: (photo_name(p), p.source))
        if not photos:
            raise StageError("This product has no photos. Import at least one.")

        outputs: dict[str, str] = {}
        rejected: list[dict[str, object]] = []
        warnings: dict[str, list[str]] = {}
        for photo in photos:
            mask, flags = mask_from_image(_read(ctx.storage, photo))
            photo.quality_flags = flags
            if mask is not None:
                png = encode_mask(mask)
                photo.mask_key = ctx.storage.put(content_key("masks", png, ".png"), png)
                outputs[f"mask/{photo.id}"] = photo.mask_key
            reasons = [REJECTING[f] for f in flags if f in REJECTING]
            if reasons:
                rejected.append(
                    {"photo": photo_name(photo), "role": photo.role, "reasons": reasons}
                )
                photo.role = PhotoRole.REJECTED.value
            elif flags:
                warnings[photo_name(photo)] = [WARNINGS[f] for f in flags]

        usable = [p for p in photos if p.role != PhotoRole.REJECTED.value]
        if not usable:
            detail = "; ".join(
                f"{photo_name(p)}: {', '.join(REJECTING[f] for f in p.quality_flags)}"
                for p in photos
            )
            raise StageError(f"No usable photos. {detail}")

        hold_out_rule = "kept from the import (or an earlier run)"
        if all(p.role is None for p in usable):
            for p in usable:
                p.role = PhotoRole.INPUT.value
            if len(usable) >= 2:
                usable[-1].role = PhotoRole.HELD_OUT.value
                hold_out_rule = "last photo by name (view labels are not built yet)"
            else:
                hold_out_rule = "none: only one usable photo, so nothing can verify accuracy"
        ctx.session.flush()

    held_out = [photo_name(p) for p in usable if p.role == PhotoRole.HELD_OUT.value]
    return StageResult(
        stage=StageName.INGEST,
        status=StageStatus.SUCCEEDED,
        metrics=meter.metrics(),
        outputs=outputs,
        summary={
            "photos": len(photos),
            "usable": len(usable),
            "held_out": held_out,
            "hold_out_rule": hold_out_rule,
            "rejected": rejected,
            "warnings": warnings,
            "masks_from": "photo transparency (automatic background removal is not built yet)",
            "view_labels": "none yet: camera fitting searches the whole orbit",
        },
    )
