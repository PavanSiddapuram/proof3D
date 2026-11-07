"""Standards section 4, unit/property level: render a known mesh from a known camera and check
that camera fitting recovers it within tolerance."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from proof3d_geometry.camera import OrbitPose, orbit_camera, rotation_angle_deg
from proof3d_geometry.camera_fit import FitSettings, fit_camera
from proof3d_geometry.raster import rasterize

# Defaults, deliberately: the test pins the configuration the pipeline ships with. At 96 px the
# tripod's 5 px arms let one case converge 20 deg off; 128 px is the documented minimum.
FAST = FitSettings()

CASES = [
    # (true pose, view label given to the fitter)
    (OrbitPose(10.0, 15.0, 2.6, roll_deg=3.0, offset_x_px=6.0, offset_y_px=-4.0), "front"),
    (OrbitPose(52.0, 22.0, 3.0, offset_x_px=-5.0), "three_quarter_left"),
    (OrbitPose(-100.0, 5.0, 2.2), "right"),
    (OrbitPose(170.0, 25.0, 2.8, roll_deg=-4.0), "back"),
]


@pytest.mark.parametrize(("true_pose", "label"), CASES)
def test_recovers_known_camera(tripod: trimesh.Trimesh, true_pose: OrbitPose, label: str) -> None:
    H, W = 120, 160  # not square, and not the fit resolution
    true_cam = orbit_camera(true_pose, W, H)
    target = rasterize(tripod.vertices, tripod.faces, true_cam).mask

    fit = fit_camera(tripod.vertices, tripod.faces, target, view_label=label, settings=FAST)

    assert fit.iou > 0.9
    assert fit.scoreable
    assert rotation_angle_deg(fit.camera.R, true_cam.R) < 5.0
    assert (fit.camera.width, fit.camera.height) == (W, H)
    # The lifted camera reproduces the target at full resolution.
    full = rasterize(tripod.vertices, tripod.faces, fit.camera).mask
    iou_full = (full & target).sum() / (full | target).sum()
    assert iou_full > 0.9


def test_symmetric_product_is_flagged_ambiguous() -> None:
    """A plain box looks the same from front and back; the fitter must say so."""
    box = trimesh.creation.box(extents=[0.6, 1.0, 0.25])
    true_cam = orbit_camera(OrbitPose(0.0, 10.0, 3.0), 96, 96)
    target = rasterize(box.vertices, box.faces, true_cam).mask
    settings = FitSettings(azimuth_range_deg=180.0, max_iter=100)
    fit = fit_camera(box.vertices, box.faces, target, view_label="front", settings=settings)
    assert fit.iou > 0.9
    assert fit.ambiguous


def test_wrong_shape_is_flagged_low_iou(tripod: trimesh.Trimesh) -> None:
    """A target the mesh cannot explain must be flagged, not scored."""
    target = np.zeros((96, 96), dtype=bool)
    yy, xx = np.mgrid[:96, :96]
    target[(yy - 48) ** 2 + (xx - 48) ** 2 < 30**2] = True  # a disc
    target[(yy - 48) ** 2 + (xx - 48) ** 2 < 12**2] = False  # with a hole
    # Only the flag is under test, so search cheaply.
    cheap = FitSettings(resolution=64, supersample=1, top_k=2, max_iter=60, polish_restarts=0)
    fit = fit_camera(tripod.vertices, tripod.faces, target, view_label="front", settings=cheap)
    assert fit.low_iou
    assert not fit.scoreable


def test_empty_mask_rejected(tripod: trimesh.Trimesh) -> None:
    with pytest.raises(ValueError, match="empty"):
        fit_camera(tripod.vertices, tripod.faces, np.zeros((32, 32), dtype=bool))
