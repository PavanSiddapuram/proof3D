"""check_model end to end on a synthetic product with known cameras."""

from __future__ import annotations

import json

import numpy as np
import pytest
import trimesh

from proof3d_geometry.camera import OrbitPose, orbit_camera
from proof3d_geometry.camera_fit import FitSettings
from proof3d_geometry.check import PhotoInput, check_model
from proof3d_geometry.mesh import cluster_decimate
from proof3d_geometry.metrics import silhouette_iou
from proof3d_geometry.raster import rasterize, silhouette

CHEAP = FitSettings(resolution=64, supersample=1, top_k=2, max_iter=60, polish_restarts=0)


def _mask(mesh: trimesh.Trimesh, az: float, el: float) -> np.ndarray:
    cam = orbit_camera(OrbitPose(az, el, 2.8), 160, 120)
    return rasterize(mesh.vertices, mesh.faces, cam).mask


def _disc_with_hole() -> np.ndarray:
    yy, xx = np.mgrid[:96, :96]
    r2 = (yy - 48) ** 2 + (xx - 48) ** 2
    return (r2 < 30**2) & (r2 >= 12**2)


def test_check_reports_fits_shares_and_suggestions(tripod: trimesh.Trimesh) -> None:
    photos = [
        PhotoInput("front.png", _mask(tripod, 0, 10), "front"),
        PhotoInput("left.png", _mask(tripod, 90, 10), "left"),
        PhotoInput("back.png", _mask(tripod, 180, 10), "back", held_out=True),
    ]
    report = check_model(tripod.vertices, tripod.faces, photos)

    assert [p.low_iou for p in report.photos] == [False, False, False]
    assert all(p.silhouette_iou > 0.9 for p in report.photos)
    back = report.photos[2]
    assert back.surface_shown > 0
    assert report.checkable_share == pytest.approx(back.surface_shown)
    assert report.evidenced_share > report.checkable_share
    assert report.invented_share == pytest.approx(1 - report.evidenced_share)
    assert report.notes == []
    assert report.held_out_disagreements == []
    assert all(s.gain > 0 for s in report.suggestions)
    json.dumps(report.to_dict())  # the CLI writes this


def test_without_held_out_photos_it_says_so(tripod: trimesh.Trimesh) -> None:
    report = check_model(
        tripod.vertices, tripod.faces, [PhotoInput("front.png", _mask(tripod, 0, 10), "front")]
    )
    assert report.checkable_share == 0.0
    assert report.evidenced_share > 0
    assert any("consistency with the inputs, not accuracy" in n for n in report.notes)


def test_bad_fits_are_left_out_and_reported(tripod: trimesh.Trimesh) -> None:
    photos = [PhotoInput("odd.png", _disc_with_hole(), "front", held_out=True)]
    report = check_model(tripod.vertices, tripod.faces, photos, fit_settings=CHEAP, max_faces=40)
    assert report.photos[0].low_iou
    assert report.photos[0].surface_shown == 0.0
    assert report.evidenced_share == 0.0
    assert report.faces_used <= 40
    assert report.held_out_disagreements == ["odd.png"]
    assert report.suggestions == []  # fix the model or mask first; more photos will not help
    notes = " ".join(report.notes)
    assert "disagrees with held-out photo(s) it never saw: odd.png" in notes
    assert "decimated from 48" in notes


def test_input_photo_that_does_not_fit_points_at_the_mask(tripod: trimesh.Trimesh) -> None:
    photos = [PhotoInput("odd.png", _disc_with_hole(), "front")]
    report = check_model(tripod.vertices, tripod.faces, photos, fit_settings=CHEAP)
    assert report.held_out_disagreements == []
    assert any("does not match its own input photo(s) odd.png" in n for n in report.notes)


def test_check_needs_photos(tripod: trimesh.Trimesh) -> None:
    with pytest.raises(ValueError, match="at least one photo"):
        check_model(tripod.vertices, tripod.faces, [])


def test_cluster_decimate_keeps_the_silhouette() -> None:
    sphere = trimesh.creation.icosphere(subdivisions=5, radius=0.5)  # 20,480 faces
    v, f = cluster_decimate(sphere.vertices, sphere.faces, 5_000)
    assert len(f) <= 5_000
    for az, el in [(0, 0), (70, 40)]:
        cam = orbit_camera(OrbitPose(az, el, 3.0), 128, 128)
        full = silhouette(sphere.vertices, sphere.faces, cam)
        assert silhouette_iou(full, silhouette(v, f, cam)) > 0.98


def test_cluster_decimate_leaves_small_meshes_alone(tripod: trimesh.Trimesh) -> None:
    v, f = cluster_decimate(tripod.vertices, tripod.faces, 1_000)
    assert np.array_equal(f, tripod.faces)
    assert np.array_equal(v, tripod.vertices)
    with pytest.raises(ValueError, match="at least 1"):
        cluster_decimate(tripod.vertices, tripod.faces, 0)
    with pytest.raises(ValueError, match="cannot decimate"):
        cluster_decimate(tripod.vertices, tripod.faces, 1)  # never returns an empty mesh
