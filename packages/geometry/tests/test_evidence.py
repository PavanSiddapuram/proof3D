from __future__ import annotations

import itertools

import numpy as np
import pytest
import trimesh

from proof3d_geometry.camera import Camera, OrbitPose, orbit_camera
from proof3d_geometry.evidence import (
    EvidenceSettings,
    SurfaceEvidence,
    face_geometry,
    suggest_views,
    surface_evidence,
    view_name,
    visible_faces,
)

AXIS_VIEWS = [(0, 0), (90, 0), (180, 0), (-90, 0), (0, 85), (0, -85)]


def _camera(az: float, el: float, distance: float = 4.0, size: int = 160) -> Camera:
    return orbit_camera(OrbitPose(az, el, distance), size, size)


def _box(extents: tuple[float, float, float] = (1.0, 0.4, 0.2)) -> trimesh.Trimesh:
    return trimesh.creation.box(extents=extents)


def _area_share(mesh: trimesh.Trimesh, seen: np.ndarray) -> float:
    return float(mesh.area_faces[seen].sum() / mesh.area)


def test_front_view_sees_only_the_front_face() -> None:
    box = _box()
    seen = visible_faces(box.vertices, box.faces, _camera(0, 0))
    # The front face (normal +z) is 1.0 x 0.4; the sides are edge-on and excluded.
    assert seen.sum() == 2
    assert (np.round(box.face_normals[seen], 6) == [0, 0, 1]).all()
    assert _area_share(box, seen) == pytest.approx(0.4 / box.area)


def test_occluded_surface_is_not_seen() -> None:
    near = trimesh.creation.box(extents=[1.0, 1.0, 0.05])
    far = near.copy()
    far.apply_translation([0, 0, -0.5])
    both = trimesh.util.concatenate([near, far])
    seen = visible_faces(both.vertices, both.faces, _camera(0, 0))
    n_near = len(near.faces)
    assert seen[:n_near].any()
    assert not seen[n_near:].any()  # the far slab is entirely behind the near one


def test_axis_views_cover_a_box() -> None:
    box = _box()
    ev = surface_evidence(box.vertices, box.faces, [_camera(a, e) for a, e in AXIS_VIEWS])
    assert ev.share() == pytest.approx(1.0)
    assert ev.share([0]) == pytest.approx(0.4 / box.area)
    assert ev.share([]) == 0.0
    assert ev.seen_by([0, 2]).sum() == 4  # front and back: two triangles each


def test_share_is_stable_across_resolution_on_a_dense_mesh() -> None:
    """Faces far smaller than a pixel still count, so a coarse depth buffer gives the same
    answer as a fine one. Counting face-ID pixels instead would fail this badly."""
    sphere = trimesh.creation.icosphere(subdivisions=5, radius=0.5)  # 20,480 faces
    cams = [_camera(0, 0), _camera(90, 30)]
    shares = [
        surface_evidence(
            sphere.vertices, sphere.faces, cams, settings=EvidenceSettings(resolution=r)
        ).share()
        for r in (48, 256)
    ]
    assert shares[0] > 0.3
    assert shares[0] == pytest.approx(shares[1], abs=0.02)


def test_view_angle_limit_excludes_oblique_faces() -> None:
    box = _box()
    cam = _camera(60, 0)  # front face at 60 degrees, right side (+x) at 30 degrees
    loose = visible_faces(box.vertices, box.faces, cam)
    strict = visible_faces(
        box.vertices, box.faces, cam, settings=EvidenceSettings(max_view_angle_deg=45)
    )
    normals = np.round(box.face_normals, 6)
    front = (normals == [0, 0, 1]).all(axis=1)
    assert loose[front].all()
    assert not strict[front].any()
    assert strict[(normals == [1, 0, 0]).all(axis=1)].all()


def test_mask_limits_evidence_to_the_product_in_the_photo() -> None:
    box = _box()
    cam = _camera(0, 0)
    empty = np.zeros((cam.height, cam.width), dtype=bool)
    assert not visible_faces(box.vertices, box.faces, cam, mask=empty).any()
    full = np.ones_like(empty)
    assert visible_faces(box.vertices, box.faces, cam, mask=full).sum() == 2
    with pytest.raises(ValueError, match="does not match"):
        visible_faces(box.vertices, box.faces, cam, mask=np.ones((3, 3), dtype=bool))


def test_evidence_validates_inputs() -> None:
    box = _box()
    with pytest.raises(ValueError, match="one mask"):
        surface_evidence(box.vertices, box.faces, [_camera(0, 0)], masks=[])
    with pytest.raises(ValueError, match="zero surface"):
        SurfaceEvidence(seen=np.zeros((1, 1), dtype=bool), areas=np.zeros(1))
    with pytest.raises(ValueError, match="match areas"):
        SurfaceEvidence(seen=np.zeros((1, 2), dtype=bool), areas=np.ones(3))
    assert surface_evidence(box.vertices, box.faces, []).share() == 0.0


def test_degenerate_faces_are_never_seen() -> None:
    v = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0]], dtype=float)
    f = np.array([[0, 1, 2], [0, 1, 3]])
    areas, normals, _ = face_geometry(v, f)
    assert areas[0] == 0
    assert not normals[0].any()
    seen = visible_faces(v, f, _camera(0, 0))
    assert not seen[0]
    assert seen[1]


@pytest.mark.parametrize(
    ("az", "el", "name"),
    [
        (0, 10, "front"),
        (170, 0, "back"),
        (-170, 0, "back"),
        (100, 20, "left"),
        (-80, 0, "right"),
        (40, 15, "three_quarter_left"),
        (315, 0, "three_quarter_right"),
        (30, 70, "top"),
        (0, -60, "underside"),
    ],
)
def test_view_name(az: float, el: float, name: str) -> None:
    assert view_name(az, el) == name


def test_suggests_the_back_after_a_front_photo() -> None:
    box = _box()
    ev = surface_evidence(box.vertices, box.faces, [_camera(0, 0)])
    suggestions = suggest_views(box.vertices, box.faces, ev.seen_by(), count=3)
    first = suggestions[0]
    assert first.view == "back"
    assert first.gain > 0.4 / box.area  # the back face, plus the top seen from above
    assert first.covered_after == pytest.approx(ev.share() + first.gain)
    gains = [s.gain for s in suggestions]
    assert gains == sorted(gains, reverse=True)
    assert all(b.covered_after > a.covered_after for a, b in itertools.pairwise(suggestions))


def test_no_suggestion_once_everything_is_seen() -> None:
    box = _box()
    everything = np.ones(len(box.faces), dtype=bool)
    assert suggest_views(box.vertices, box.faces, everything) == []


def test_underside_is_only_suggested_when_asked() -> None:
    box = _box()
    ev = surface_evidence(
        box.vertices, box.faces, [_camera(a, e) for a, e in AXIS_VIEWS[:5]]
    )  # every side but the bottom
    assert suggest_views(box.vertices, box.faces, ev.seen_by()) == []
    below = suggest_views(box.vertices, box.faces, ev.seen_by(), elevations=(-60.0,))
    assert below[0].view == "underside"
    assert below[0].covered_after == pytest.approx(1.0)
