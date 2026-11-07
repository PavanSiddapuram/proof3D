from __future__ import annotations

import numpy as np
import pytest
import trimesh
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays

from proof3d_geometry.camera import OrbitPose, orbit_camera
from proof3d_geometry.mesh import check_mesh, normalize
from proof3d_geometry.metrics import agreement_mask, silhouette_iou
from proof3d_geometry.raster import rasterize
from proof3d_geometry.regions import face_adjacency, failing_regions

masks = arrays(np.bool_, (12, 12))


@settings(max_examples=200, deadline=None)
@given(masks, masks)
def test_iou_bounds_and_symmetry(a: np.ndarray, b: np.ndarray) -> None:
    v = silhouette_iou(a, b)
    assert 0.0 <= v <= 1.0
    assert v == silhouette_iou(b, a)
    assert silhouette_iou(a, a) == 1.0


def test_iou_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shapes differ"):
        silhouette_iou(np.zeros((2, 2), bool), np.zeros((3, 3), bool))


def test_agreement_mask_is_eroded_intersection() -> None:
    a = np.zeros((20, 20), bool)
    a[2:18, 2:18] = True
    b = np.zeros((20, 20), bool)
    b[5:20, 5:20] = True
    m = agreement_mask(a, b, erode_px=1)
    assert m.sum() < (a & b).sum()
    assert not (m & ~(a & b)).any()


def test_face_adjacency_of_cube(unit_cube: trimesh.Trimesh) -> None:
    pairs = face_adjacency(unit_cube.faces)
    assert len(pairs) == 18  # 12 triangles, 18 interior/shared edges on a closed cube


def test_failing_region_found_on_rendered_mesh(unit_cube: trimesh.Trimesh) -> None:
    cam = orbit_camera(OrbitPose(0.0, 0.0, 3.0), 64, 64)
    out = rasterize(unit_cube.vertices, unit_cube.faces, cam)
    visible = np.unique(out.face_id[out.mask])
    bad_faces = set(visible.tolist())  # the two front-facing triangles
    error = np.where(np.isin(out.face_id, list(bad_faces)), 0.9, 0.05)
    regions = failing_regions(out.face_id, error, out.mask, unit_cube.faces, threshold=0.5)
    assert len(regions) == 1
    assert set(regions[0].face_ids) == bad_faces
    assert regions[0].mean_error == pytest.approx(0.9)


def test_disconnected_failures_are_separate_regions(tripod: trimesh.Trimesh) -> None:
    cam = orbit_camera(OrbitPose(30.0, 20.0, 2.5), 128, 128)
    out = rasterize(tripod.vertices, tripod.faces, cam)
    part = np.repeat(np.arange(4), 12)  # four boxes, 12 triangles each
    bad = np.isin(part[np.maximum(out.face_id, 0)], [1, 2]) & out.mask  # two separate arms
    error = np.where(bad, 1.0, 0.0)
    regions = failing_regions(out.face_id, error, out.mask, tripod.faces, threshold=0.5)
    assert len(regions) == 2
    for r in regions:
        assert len({int(part[f]) for f in r.face_ids}) == 1


def test_no_failures() -> None:
    fid = np.zeros((4, 4), np.int32)
    faces = np.array([[0, 1, 2]])
    assert failing_regions(fid, np.zeros((4, 4)), np.ones((4, 4), bool), faces, 0.5) == []


def test_normalize() -> None:
    v = np.array([[0, 0, 0], [4, 2, 1], [1, 1, 1]], dtype=float)
    n, centre, scale = normalize(v)
    np.testing.assert_allclose(n.max(axis=0) + n.min(axis=0), 0, atol=1e-12)
    assert (n.max(axis=0) - n.min(axis=0)).max() == pytest.approx(1.0)
    np.testing.assert_allclose(n / scale + centre, v)


def test_check_mesh_clean_and_broken(unit_cube: trimesh.Trimesh) -> None:
    clean = check_mesh(unit_cube)
    assert clean.watertight
    assert not clean.needs_geometry_repair
    broken = trimesh.Trimesh(
        unit_cube.vertices,
        np.vstack([unit_cube.faces, unit_cube.faces[:1]]),
        process=False,
    )
    report = check_mesh(broken)
    assert report.duplicate_faces == 1
    assert report.needs_geometry_repair
