from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from proof3d_geometry.camera import OrbitPose, look_at_rotation, orbit_camera, rotation_angle_deg

az = st.floats(-180, 180)
el = st.floats(-80, 80)
roll = st.floats(-45, 45)
dist = st.floats(0.5, 10)
fov = st.floats(15, 80)


@settings(max_examples=200, deadline=None)
@given(az, el, roll, dist, fov)
def test_origin_projects_to_principal_point(
    a: float, e: float, r: float, d: float, f: float
) -> None:
    cam = orbit_camera(OrbitPose(a, e, d, roll_deg=r, fov_deg=f), 200, 100)
    uv, z = cam.project(np.zeros((1, 3)))
    np.testing.assert_allclose(uv[0], [cam.cx, cam.cy], atol=1e-9)
    assert z[0] == pytest.approx(d)


@settings(max_examples=200, deadline=None)
@given(az, el, roll, dist)
def test_rotation_is_orthonormal_and_centre_is_on_orbit(
    a: float, e: float, r: float, d: float
) -> None:
    cam = orbit_camera(OrbitPose(a, e, d, roll_deg=r), 64, 64)
    np.testing.assert_allclose(cam.R @ cam.R.T, np.eye(3), atol=1e-9)
    assert np.linalg.det(cam.R) == pytest.approx(1.0)
    assert np.linalg.norm(cam.center) == pytest.approx(d)


@settings(max_examples=100, deadline=None)
@given(az, el, dist, st.floats(0.25, 4.0))
def test_scaled_camera_scales_pixel_coordinates(a: float, e: float, d: float, k: float) -> None:
    cam = orbit_camera(OrbitPose(a, e, d + 2.0), 128, 96)
    pts = np.array([[0.1, 0.2, -0.3], [-0.4, 0.0, 0.25]])
    uv, _ = cam.project(pts)
    uv_k, _ = cam.scaled(k).project(pts)
    np.testing.assert_allclose(uv_k, uv * k, rtol=1e-9, atol=1e-9)


def test_front_view_conventions() -> None:
    """Front camera sits on +z; world +x appears to the image right, world +y appears up."""
    cam = orbit_camera(OrbitPose(0.0, 0.0, 3.0), 100, 100)
    np.testing.assert_allclose(cam.center, [0, 0, 3], atol=1e-12)
    uv, _ = cam.project(np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]]))
    assert uv[0, 0] > cam.cx  # +x to the right
    assert uv[1, 1] < cam.cy  # +y up (image y points down)


def test_left_view_sees_positive_x_side() -> None:
    cam = orbit_camera(OrbitPose(90.0, 0.0, 3.0), 64, 64)
    np.testing.assert_allclose(cam.center, [3, 0, 0], atol=1e-12)


def test_look_at_singular_straight_down() -> None:
    with pytest.raises(ValueError, match="singular"):
        look_at_rotation(np.array([0.0, 3.0, 0.0]), np.zeros(3))


def test_rotation_angle() -> None:
    a = orbit_camera(OrbitPose(0.0, 0.0, 3.0), 8, 8).R
    b = orbit_camera(OrbitPose(30.0, 0.0, 3.0), 8, 8).R
    assert rotation_angle_deg(a, a) == pytest.approx(0.0, abs=1e-6)
    assert rotation_angle_deg(a, b) == pytest.approx(30.0, abs=1e-6)
