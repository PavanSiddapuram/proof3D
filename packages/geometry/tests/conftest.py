from __future__ import annotations

import numpy as np
import pytest
import trimesh

from proof3d_geometry.mesh import normalize


def tripod_mesh() -> trimesh.Trimesh:
    """Chiral test product: a base block with three arms of different lengths, so every
    viewpoint gives a distinct silhouette and pose recovery is well-posed."""
    parts = [
        trimesh.creation.box(extents=[0.30, 0.30, 0.30]),
        trimesh.creation.box(extents=[1.00, 0.12, 0.12]),
        trimesh.creation.box(extents=[0.12, 0.60, 0.12]),
        trimesh.creation.box(extents=[0.12, 0.12, 0.35]),
    ]
    offsets = [[0, 0, 0], [0.5, 0, 0], [0, 0.3, 0], [0, 0, 0.175]]
    for part, off in zip(parts, offsets, strict=True):
        part.apply_translation(off)
    mesh = trimesh.util.concatenate(parts)
    v, _, _ = normalize(mesh.vertices)
    return trimesh.Trimesh(v, mesh.faces, process=False)


@pytest.fixture(scope="session")
def tripod() -> trimesh.Trimesh:
    return tripod_mesh()


@pytest.fixture(scope="session")
def unit_cube() -> trimesh.Trimesh:
    return trimesh.creation.box(extents=[1.0, 1.0, 1.0])


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
