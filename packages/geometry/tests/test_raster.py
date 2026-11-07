from __future__ import annotations

import numpy as np
import pytest
import trimesh
from hypothesis import given, settings
from hypothesis import strategies as st

from proof3d_geometry.camera import Camera, OrbitPose, orbit_camera
from proof3d_geometry.raster import rasterize, rasterize_reference


def _ortho_like_camera(size: int = 64) -> Camera:
    # Long focal length from far away: close to orthographic, easy to reason about.
    return orbit_camera(OrbitPose(0.0, 0.0, 50.0, fov_deg=2.0), size, size)


def test_empty_when_nothing_in_view(unit_cube: trimesh.Trimesh) -> None:
    cam = orbit_camera(OrbitPose(0.0, 0.0, 3.0), 32, 32)
    shifted = unit_cube.vertices + np.array([0.0, 0.0, 10.0])  # behind the camera
    out = rasterize(shifted, unit_cube.faces, cam)
    assert not out.mask.any()
    assert np.isinf(out.depth).all()


def test_square_face_on_area(unit_cube: trimesh.Trimesh) -> None:
    cam = _ortho_like_camera(64)
    out = rasterize(unit_cube.vertices, unit_cube.faces, cam)
    # Cube front face spans f * 1 / ~49.5 px on a side.
    side = cam.fx * 1.0 / 49.5
    assert out.mask.sum() == pytest.approx(side * side, rel=0.08)


def test_nearest_face_wins() -> None:
    near = np.array([[-1, -1, 0.5], [1, -1, 0.5], [0, 1, 0.5]], dtype=float)
    far = near.copy()
    far[:, 2] = -0.5
    verts = np.vstack([far, near])
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    cam = orbit_camera(OrbitPose(0.0, 0.0, 5.0), 48, 48)
    out = rasterize(verts, faces, cam)
    assert set(np.unique(out.face_id)) == {-1, 1}
    assert np.nanmin(out.depth[out.mask]) == pytest.approx(4.5, abs=1e-6)


def test_winding_does_not_change_silhouette(unit_cube: trimesh.Trimesh) -> None:
    cam = orbit_camera(OrbitPose(30.0, 20.0, 3.0), 64, 64)
    a = rasterize(unit_cube.vertices, unit_cube.faces, cam).mask
    b = rasterize(unit_cube.vertices, unit_cube.faces[:, ::-1], cam).mask
    np.testing.assert_array_equal(a, b)


@settings(max_examples=25, deadline=None)
@given(st.floats(-180, 180), st.floats(-60, 60), st.floats(2.0, 4.0), st.floats(1.1, 2.0))
def test_area_shrinks_with_distance(a: float, e: float, d: float, k: float) -> None:
    box = trimesh.creation.box(extents=[1.0, 0.6, 0.3])
    near = rasterize(box.vertices, box.faces, orbit_camera(OrbitPose(a, e, d), 64, 64)).mask
    far = rasterize(box.vertices, box.faces, orbit_camera(OrbitPose(a, e, d * k), 64, 64)).mask
    assert far.sum() <= near.sum()


@settings(max_examples=40, deadline=None)
@given(
    st.integers(0, 2**31 - 1),
    st.integers(1, 40),
    st.floats(-180, 180),
    st.floats(-70, 70),
    st.floats(1.5, 4.0),
)
def test_vectorised_matches_reference(
    seed: int, n_faces: int, a: float, e: float, d: float
) -> None:
    """Random triangle soups, some partly off-screen: both rasterisers must agree exactly."""
    rng = np.random.default_rng(seed)
    verts = rng.uniform(-0.8, 0.8, size=(n_faces * 3, 3))
    faces = np.arange(n_faces * 3).reshape(n_faces, 3)
    cam = orbit_camera(OrbitPose(a, e, d, offset_x_px=float(rng.uniform(-20, 20))), 48, 40)
    fast = rasterize(verts, faces, cam)
    ref = rasterize_reference(verts, faces, cam)
    np.testing.assert_array_equal(fast.mask, ref.mask)
    np.testing.assert_allclose(fast.depth[fast.mask], ref.depth[ref.mask], rtol=1e-12)
    # Face IDs may differ only where two faces are at the same depth.
    differ = fast.face_id != ref.face_id
    assert np.allclose(fast.depth[differ], ref.depth[differ])
