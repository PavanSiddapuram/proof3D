"""Check any 3D model against product photos, on CPU, with the parts built so far.

``check_model`` takes a mesh from any generator (VersaAI, Tripo, Rodin, ours) and the
product's photos with their masks. For each photo it fits a camera. It then reports:

* how well the silhouette matches from each photo's camera. For a held-out photo, one the
  generator never saw, that is evidence of accuracy. For an input photo it only shows the
  model is consistent with what it was built from;
* held-out disagreements: held-out photos whose silhouette the model cannot match. That is
  the main finding when present, not a fitting failure to hide;
* surface shares: shown by a held-out photo that fits (checkable), shown by any photo that
  fits (evidenced), and shown by none (invented);
* the next photos to take, to shrink the invented share. None while a held-out photo
  disagrees: the model (or that photo's mask) needs fixing first, and a suggestion could name
  a view the merchant already has, the one that disagrees.

Appearance scores (LPIPS, DINOv2, FLIP) need the GPU worker and are not part of this yet,
so the only accuracy number here is the held-out silhouette IoU.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from proof3d_geometry.camera import FloatArray
from proof3d_geometry.camera_fit import FitSettings, fit_camera
from proof3d_geometry.evidence import (
    EvidenceSettings,
    ViewSuggestion,
    suggest_views,
    surface_evidence,
)
from proof3d_geometry.mesh import cluster_decimate, normalize
from proof3d_geometry.raster import BoolArray, IntArray


@dataclass(frozen=True)
class PhotoInput:
    name: str
    mask: BoolArray  # product pixels in the photo
    view_label: str | None = None  # front, back, ... ; None searches the whole orbit
    held_out: bool = False  # True if the generator never saw this photo


@dataclass(frozen=True)
class PhotoCheck:
    name: str
    view_label: str | None
    held_out: bool
    silhouette_iou: float
    low_iou: bool  # fit too poor to use: excluded from evidence
    ambiguous: bool  # another pose fits almost as well: evidence is uncertain
    azimuth_deg: float
    elevation_deg: float
    surface_shown: float  # share of the surface this photo shows; 0 when low_iou


@dataclass(frozen=True)
class ModelCheck:
    photos: list[PhotoCheck]
    checkable_share: float  # shown by a usable held-out photo
    evidenced_share: float  # shown by any usable photo
    invented_share: float  # shown by no usable photo
    suggestions: list[ViewSuggestion]
    faces_used: int  # after decimation, if the mesh was large
    # Held-out photos whose silhouette the model cannot match: the main finding when present.
    held_out_disagreements: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_model(
    vertices: FloatArray,
    faces: IntArray,
    photos: list[PhotoInput],
    fit_settings: FitSettings | None = None,
    evidence_settings: EvidenceSettings | None = None,
    max_faces: int = 20_000,
    suggestions: int = 3,
) -> ModelCheck:
    if not photos:
        raise ValueError("give at least one photo")
    v, _, _ = normalize(vertices)
    v, f = cluster_decimate(v, np.asarray(faces, dtype=np.int64), max_faces)

    fits = [fit_camera(v, f, p.mask, p.view_label, fit_settings) for p in photos]
    usable = [i for i, fit in enumerate(fits) if not fit.low_iou]
    evidence = surface_evidence(
        v, f, [fits[i].camera for i in usable], [photos[i].mask for i in usable], evidence_settings
    )
    row = {photo_index: r for r, photo_index in enumerate(usable)}
    held_rows = [row[i] for i in usable if photos[i].held_out]

    checks = [
        PhotoCheck(
            name=p.name,
            view_label=p.view_label,
            held_out=p.held_out,
            silhouette_iou=float(fit.iou),
            low_iou=bool(fit.low_iou),
            ambiguous=bool(fit.ambiguous),
            azimuth_deg=float(fit.pose.azimuth_deg),
            elevation_deg=float(fit.pose.elevation_deg),
            surface_shown=evidence.share([row[i]]) if i in row else 0.0,
        )
        for i, (p, fit) in enumerate(zip(photos, fits, strict=True))
    ]

    disagreements = [p for p in checks if p.held_out and p.low_iou]
    notes = []
    if disagreements:
        listed = ", ".join(f"{p.name} (IoU {p.silhouette_iou:.2f})" for p in disagreements)
        notes.append(
            f"The model disagrees with held-out photo(s) it never saw: {listed}. Either the "
            "model is wrong from those views or the photo's mask is; check the mask first."
        )
    if not any(p.held_out for p in photos):
        notes.append(
            "No held-out photo: every photo was seen by the generator, so this shows "
            "consistency with the inputs, not accuracy."
        )
    inputs_failed = [p.name for p in checks if not p.held_out and p.low_iou]
    if inputs_failed:
        notes.append(
            f"The model does not match its own input photo(s) {', '.join(inputs_failed)}. "
            "Check their masks and view labels (or pass none, if the model's front is not +z)."
        )
    if len(f) < len(faces):
        notes.append(f"Mesh decimated from {len(faces)} to {len(f)} faces for checking.")

    evidenced = evidence.share()
    return ModelCheck(
        photos=checks,
        checkable_share=evidence.share(held_rows),
        evidenced_share=evidenced,
        invented_share=1.0 - evidenced,
        suggestions=[]
        if disagreements
        else suggest_views(v, f, evidence.seen_by(), count=suggestions, settings=evidence_settings),
        faces_used=len(f),
        held_out_disagreements=[p.name for p in disagreements],
        notes=notes,
    )
