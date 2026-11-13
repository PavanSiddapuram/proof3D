"""Surface evidence: which parts of a 3D model any photo actually shows.

A verifier can only check what some photo shows. For each photo with a fitted camera, this
module marks the faces that photo sees clearly. The scorecard then reports shares of the
surface, by area:

* **verified**: seen by a held-out photo, so the score covers it;
* **evidenced**: seen by any photo, input or held-out;
* **invented**: seen by no photo, so the generator guessed it.

``suggest_views`` turns the invented share into a request a merchant can act on: the next
photo that would show the most unseen surface.

Visibility is a depth test at each face's centroid, not a count of face-ID pixels. A dense
mesh has many faces smaller than a pixel, and counting pixels would call most of them unseen.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from proof3d_geometry.camera import Camera, FloatArray, OrbitPose, orbit_camera
from proof3d_geometry.raster import NEAR, BoolArray, IntArray, RasterOutput, rasterize

Rasterizer = Callable[[FloatArray, IntArray, Camera], RasterOutput]


@dataclass(frozen=True)
class EvidenceSettings:
    resolution: int = 256  # long side of the depth buffer used for occlusion
    # A face seen more obliquely than this is not evidence: a photo says little about a
    # surface it sees almost edge-on. A default to tune against the human ratings.
    max_view_angle_deg: float = 75.0
    # Occlusion slack in mesh units, on top of the pixel-size slack below. Meshes are
    # normalised to 1 across, so this is 1% of size.
    depth_tolerance: float = 0.01


@dataclass(frozen=True)
class SurfaceEvidence:
    seen: BoolArray  # (n_photos, n_faces): photo i shows face j clearly
    areas: FloatArray  # (n_faces,)

    def __post_init__(self) -> None:
        if self.seen.ndim != 2 or self.seen.shape[1] != self.areas.shape[0]:
            raise ValueError("seen must be (n_photos, n_faces) and match areas")
        if self.areas.sum() <= 0:
            raise ValueError("mesh has zero surface area")

    def seen_by(self, photos: Sequence[int] | None = None) -> BoolArray:
        """Faces shown by any of ``photos`` (all photos when None)."""
        rows = self.seen if photos is None else self.seen[list(photos)]
        return np.asarray(rows.any(axis=0)) if len(rows) else np.zeros_like(self.areas, bool)

    def share(self, photos: Sequence[int] | None = None) -> float:
        """Share of the surface area shown by any of ``photos`` (all photos when None)."""
        return float(self.areas[self.seen_by(photos)].sum() / self.areas.sum())


@dataclass(frozen=True)
class ViewSuggestion:
    view: str  # front, back, left, right, three_quarter_left/right, top, underside
    azimuth_deg: float
    elevation_deg: float
    gain: float  # share of the surface this photo would newly show
    covered_after: float  # share shown by some photo once this and earlier suggestions exist


def face_geometry(vertices: FloatArray, faces: IntArray) -> tuple[FloatArray, ...]:
    """Per-face ``(areas, unit normals, centroids)``. Degenerate faces get a zero normal."""
    tri = np.asarray(vertices, dtype=np.float64)[np.asarray(faces)]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(cross, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        normals = np.where(norm[:, None] > 0, cross / norm[:, None], 0.0)
    return 0.5 * norm, normals, tri.mean(axis=1)


def visible_faces(
    vertices: FloatArray,
    faces: IntArray,
    camera: Camera,
    mask: BoolArray | None = None,
    settings: EvidenceSettings | None = None,
    rasterizer: Rasterizer = rasterize,
) -> BoolArray:
    """Faces this camera sees clearly: in frame, unoccluded, not too oblique.

    ``mask`` is the photo's product mask at the camera's resolution. When given, a face counts
    only if it lands on the product in the photo, so a loose camera fit cannot claim evidence
    for surface the photo does not actually show.
    """
    return _visibility(vertices, faces, camera, mask, settings, rasterizer)[0]


def _visibility(
    vertices: FloatArray,
    faces: IntArray,
    camera: Camera,
    mask: BoolArray | None,
    settings: EvidenceSettings | None,
    rasterizer: Rasterizer,
) -> tuple[BoolArray, FloatArray]:
    """``(seen, cos)``: the visible faces, and the cosine of each face's viewing angle."""
    s = settings or EvidenceSettings()
    if mask is not None and mask.shape != (camera.height, camera.width):
        raise ValueError(f"mask {mask.shape} does not match camera {camera.height, camera.width}")
    areas, normals, centroids = face_geometry(vertices, faces)
    scale = s.resolution / max(camera.width, camera.height)
    cam = camera.scaled(scale)
    depth = rasterizer(vertices, faces, cam).depth

    uv, z = cam.project(centroids)
    finite = np.isfinite(uv).all(axis=1) & (z > NEAR)
    px = np.floor(np.where(finite, uv[:, 0], -1)).astype(np.int64)
    py = np.floor(np.where(finite, uv[:, 1], -1)).astype(np.int64)
    inside = finite & (px >= 0) & (px < cam.width) & (py >= 0) & (py < cam.height)
    pxc, pyc = np.clip(px, 0, cam.width - 1), np.clip(py, 0, cam.height - 1)

    # The depth buffer is sampled at pixel centres, but a centroid can sit up to ~0.7 px away.
    # Across that offset, a surface tilted at the steepest allowed angle changes depth by about
    # one pixel's footprint times tan(angle). Without this slack, a coarse buffer hides
    # visible faces on curved surfaces.
    footprint = np.abs(z) / cam.fy
    slope = math.tan(math.radians(min(s.max_view_angle_deg, 85.0)))
    surface = depth[pyc, pxc]
    unoccluded = np.isfinite(surface) & (z <= surface + s.depth_tolerance + footprint * slope)

    to_camera = cam.center - centroids
    dist = np.linalg.norm(to_camera, axis=1)
    cos = np.abs((normals * to_camera).sum(axis=1)) / np.maximum(dist, 1e-12)
    facing = cos >= math.cos(math.radians(s.max_view_angle_deg))

    seen = inside & unoccluded & facing & (areas > 0)
    if mask is not None:
        mx = np.clip(np.floor(uv[:, 0] / scale), 0, camera.width - 1)
        my = np.clip(np.floor(uv[:, 1] / scale), 0, camera.height - 1)
        mx = np.where(finite, mx, 0).astype(np.int64)
        my = np.where(finite, my, 0).astype(np.int64)
        seen &= mask[my, mx]
    return np.asarray(seen), np.asarray(cos)


def surface_evidence(
    vertices: FloatArray,
    faces: IntArray,
    cameras: Sequence[Camera],
    masks: Sequence[BoolArray | None] | None = None,
    settings: EvidenceSettings | None = None,
    rasterizer: Rasterizer = rasterize,
) -> SurfaceEvidence:
    """Which faces each photo shows. ``cameras[i]`` is photo i's fitted camera."""
    if masks is not None and len(masks) != len(cameras):
        raise ValueError("give one mask (or None) per camera")
    areas = face_geometry(vertices, faces)[0]
    rows = [
        visible_faces(
            vertices, faces, cam, None if masks is None else masks[i], settings, rasterizer
        )
        for i, cam in enumerate(cameras)
    ]
    seen = np.stack(rows) if rows else np.zeros((0, len(areas)), dtype=bool)
    return SurfaceEvidence(seen=seen, areas=areas)


_HORIZONTAL_VIEWS = {
    "front": 0.0,
    "three_quarter_left": 45.0,
    "left": 90.0,
    "back": 180.0,
    "right": -90.0,
    "three_quarter_right": -45.0,
}


def view_name(azimuth_deg: float, elevation_deg: float) -> str:
    """Name a camera position the way a merchant would (conventions: docs/architecture.md)."""
    if elevation_deg >= 60:
        return "top"
    if elevation_deg <= -45:
        return "underside"

    def gap(a: float, b: float) -> float:
        return abs((a - b + 180.0) % 360.0 - 180.0)

    return min(_HORIZONTAL_VIEWS, key=lambda name: gap(azimuth_deg, _HORIZONTAL_VIEWS[name]))


def suggest_views(
    vertices: FloatArray,
    faces: IntArray,
    seen: BoolArray,
    count: int = 3,
    min_gain: float = 0.02,
    azimuths: Sequence[float] = tuple(range(0, 360, 30)),
    elevations: Sequence[float] = (0.0, 30.0, 60.0, 85.0),
    fov_deg: float = 40.0,
    settings: EvidenceSettings | None = None,
    rasterizer: Rasterizer = rasterize,
) -> list[ViewSuggestion]:
    """The next photos to take, greedily: each shows the most unseen surface, most head-on.

    ``seen`` is the per-face evidence so far (``SurfaceEvidence.seen_by()``). Candidates are
    ranked by newly shown area weighted by the cosine of the viewing angle, so a camera that
    sees the back square-on beats one that catches it at a glancing angle. ``gain`` reports
    the plain area share. Suggestions stop when the best candidate would add less than
    ``min_gain`` of the surface. The default
    elevations leave out the underside, because most products are photographed resting on it;
    pass negative elevations for products where the underside matters, such as shoe soles.
    """
    s = settings or EvidenceSettings()
    areas = face_geometry(vertices, faces)[0]
    total = float(areas.sum())
    if total <= 0:
        raise ValueError("mesh has zero surface area")
    radius = float(np.linalg.norm(np.asarray(vertices, dtype=np.float64), axis=1).max())
    distance = 1.1 * radius / math.sin(math.radians(fov_deg) / 2.0)

    poses = [(float(a), float(e)) for e in elevations for a in azimuths]
    visibility = [
        _visibility(
            vertices,
            faces,
            orbit_camera(OrbitPose(a, e, distance, fov_deg=fov_deg), s.resolution, s.resolution),
            None,
            s,
            rasterizer,
        )
        for a, e in poses
    ]
    candidates = np.stack([v[0] for v in visibility])
    quality = np.stack([v[0] * v[1] for v in visibility])  # cos where seen, else 0

    covered = np.asarray(seen, dtype=bool).copy()
    suggestions: list[ViewSuggestion] = []
    for _ in range(count):
        new = ~covered
        best = int(np.argmax((quality * new) @ areas))
        gain = float(candidates[best] @ (areas * new) / total)
        if gain < min_gain:
            break
        covered |= candidates[best]
        a, e = poses[best]
        suggestions.append(
            ViewSuggestion(
                view=view_name(a, e),
                azimuth_deg=a,
                elevation_deg=e,
                gain=gain,
                covered_after=float(areas[covered].sum() / total),
            )
        )
    return suggestions
