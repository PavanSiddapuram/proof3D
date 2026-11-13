from __future__ import annotations

import numpy as np
import pytest
import trimesh

from proof3d_geometry.synthetic import tripod_mesh


@pytest.fixture(scope="session")
def tripod() -> trimesh.Trimesh:
    return tripod_mesh()


@pytest.fixture(scope="session")
def unit_cube() -> trimesh.Trimesh:
    return trimesh.creation.box(extents=[1.0, 1.0, 1.0])


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
