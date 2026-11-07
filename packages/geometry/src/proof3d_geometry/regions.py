"""Locate failures on the mesh (PRD 5.4 step 4).

Render a face-ID buffer from the fitted camera, find faces whose visible pixels are mostly
high-error, then group those faces into connected regions over the mesh's face adjacency so
repair can target each region separately.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from proof3d_geometry.raster import IntArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class FailingRegion:
    face_ids: tuple[int, ...]
    pixel_count: int
    mean_error: float


def face_adjacency(faces: IntArray) -> NDArray[np.int64]:
    """Pairs of faces sharing an edge, shape (E, 2)."""
    faces = np.asarray(faces, dtype=np.int64)
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    owner = np.tile(np.arange(len(faces)), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges, owner = edges[order], owner[order]
    same = np.all(edges[1:] == edges[:-1], axis=1)
    pairs = np.stack([owner[:-1][same], owner[1:][same]], axis=1)
    return np.asarray(pairs[pairs[:, 0] != pairs[:, 1]], dtype=np.int64)


def failing_regions(
    face_id: NDArray[np.int32],
    error: FloatArray,
    valid: BoolArray,
    faces: IntArray,
    threshold: float,
    min_face_pixels: int = 4,
    min_bad_share: float = 0.5,
) -> list[FailingRegion]:
    """Group high-error faces into connected regions, largest first.

    ``error`` is a per-pixel error map in [0, 1]; ``valid`` restricts it to pixels where the
    silhouettes agree, so misalignment at the boundary is never blamed on the surface.
    """
    n_faces = len(faces)
    use = valid & (face_id >= 0)
    ids = face_id[use].astype(np.int64)
    err = error[use]
    pixels = np.bincount(ids, minlength=n_faces)
    bad = np.bincount(ids, weights=(err > threshold).astype(np.float64), minlength=n_faces)
    err_sum = np.bincount(ids, weights=err, minlength=n_faces)

    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(pixels > 0, bad / pixels, 0.0)
    failing = (pixels >= min_face_pixels) & (share >= min_bad_share)
    if not failing.any():
        return []

    pairs = face_adjacency(faces)
    keep = failing[pairs[:, 0]] & failing[pairs[:, 1]]
    pairs = pairs[keep]
    graph = sparse.coo_matrix(
        (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n_faces, n_faces)
    )
    _, labels = connected_components(graph, directed=False)

    regions = []
    for label in np.unique(labels[failing]):
        members = np.flatnonzero(failing & (labels == label))
        px = int(pixels[members].sum())
        regions.append(
            FailingRegion(
                face_ids=tuple(int(m) for m in members),
                pixel_count=px,
                mean_error=float(err_sum[members].sum() / px),
            )
        )
    regions.sort(key=lambda r: r.pixel_count, reverse=True)
    return regions
