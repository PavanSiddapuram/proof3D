"""Fit a camera to a photo by matching the mesh silhouette to the photo's mask (PRD 5.4 step 1).

Method
------
1. Resize the target mask so its long side is ``cfg.resolution`` px.
2. Coarse search: for each (azimuth, elevation) on a grid around the view-label prior, set
   distance and principal-point offset in closed form so the rendered silhouette matches the
   mask's area and centroid, then score hard IoU.
3. Refine the best ``top_k`` *distinct* hypotheses with Nelder-Mead on a soft IoU between
   pixel *coverage* maps: the render is supersampled and box-filtered, the photo mask is
   anti-aliased when resized. A binary silhouette is piecewise constant in the camera
   parameters and stalls a simplex search; coverage changes continuously.
4. Polish the best hypothesis with shrinking Nelder-Mead restarts.
5. Report the best fit, and flag it:
   * ``low_iou``  -- best IoU below ``min_iou``: the fit is not trusted, so it is not scored.
   * ``ambiguous`` -- a pose far from the best (> ``ambiguity_angle_deg``) scores within
     ``ambiguity_iou_margin``. Typical for symmetric products (PRD 9, first risk).

Focal length and distance are nearly degenerate for silhouettes; the FOV is held at the config
value unless ``refine_fov`` is set.

On the GPU worker the same objective is minimised by gradient descent through nvdiffrast; this
CPU path is the reference implementation and the one CI tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage, optimize

from proof3d_geometry.camera import (
    MAX_ELEVATION_DEG,
    Camera,
    FloatArray,
    OrbitPose,
    rotation_angle_deg,
)
from proof3d_geometry.metrics import blur_mask, mask_centroid_and_area, silhouette_iou, soft_iou
from proof3d_geometry.raster import IntArray, Rasterizer, rasterize
from proof3d_geometry.views import prior_for

BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class FitSettings:
    """Mirror of ``proof3d_pipeline.config.CameraFitConfig``; kept separate so the geometry
    package has no dependency on the pipeline package."""

    resolution: int = 128
    fov_deg: float = 40.0
    refine_fov: bool = False
    azimuth_range_deg: float = 60.0
    azimuth_step_deg: float = 15.0
    elevation_range_deg: float = 30.0
    elevation_step_deg: float = 10.0
    top_k: int = 3
    max_iter: int = 200
    blur_sigma_px: float = 1.0
    supersample: int = 2
    polish_restarts: int = 2
    min_iou: float = 0.85
    ambiguity_iou_margin: float = 0.02
    ambiguity_angle_deg: float = 25.0


@dataclass(frozen=True)
class Hypothesis:
    pose: OrbitPose
    iou: float


@dataclass(frozen=True)
class FitResult:
    pose: OrbitPose  # at fit resolution
    camera: Camera  # lifted to the original mask resolution
    iou: float  # hard IoU at fit resolution
    low_iou: bool
    ambiguous: bool
    alternatives: tuple[Hypothesis, ...] = field(default_factory=tuple)
    evaluations: int = 0

    @property
    def scoreable(self) -> bool:
        return not self.low_iou


def _resize_coverage(mask: BoolArray, resolution: int) -> tuple[FloatArray, float]:
    """Resize a mask so its long side is ``resolution``, returning anti-aliased coverage."""
    H, W = mask.shape
    scale = resolution / max(H, W)
    m = mask.astype(np.float64)
    if math.isclose(scale, 1.0):
        return m, 1.0
    if scale < 1.0:
        m = ndimage.gaussian_filter(m, 0.5 / scale)  # anti-alias before decimating
    out_h, out_w = max(1, round(H * scale)), max(1, round(W * scale))
    zoomed = ndimage.zoom(m, (out_h / H, out_w / W), order=1, grid_mode=True, mode="nearest")
    return np.clip(zoomed, 0.0, 1.0), scale


def _box_downsample(mask: BoolArray, k: int) -> FloatArray:
    H, W = mask.shape
    return np.asarray(mask.reshape(H // k, k, W // k, k).mean(axis=(1, 3)), dtype=np.float64)


class _Objective:
    def __init__(
        self,
        vertices: FloatArray,
        faces: IntArray,
        target_coverage: FloatArray,
        settings: FitSettings,
        rasterizer: Rasterizer,
    ) -> None:
        self.vertices, self.faces = vertices, faces
        self.target = target_coverage >= 0.5
        target = self.target
        self.H, self.W = target.shape
        self.settings = settings
        self.raster = rasterizer
        self.target_soft = blur_mask(target_coverage, settings.blur_sigma_px)
        self.target_centroid, self.target_area = mask_centroid_and_area(target)
        self.evaluations = 0
        # Photos are taken from outside the product; keeping the camera out also keeps every
        # triangle in front of the near plane and stops degenerate close-up "fits".
        self.min_distance = 1.5 * _bounding_radius(vertices)

    def render(self, pose: OrbitPose) -> BoolArray:
        self.evaluations += 1
        return self.raster(self.vertices, self.faces, pose.camera(self.W, self.H)).mask

    def coverage(self, pose: OrbitPose) -> FloatArray:
        k = self.settings.supersample
        if k <= 1:
            return self.render(pose).astype(np.float64)
        self.evaluations += 1
        cam = pose.camera(self.W, self.H).scaled(k)
        return _box_downsample(self.raster(self.vertices, self.faces, cam).mask, k)

    def hard_iou(self, pose: OrbitPose) -> float:
        return silhouette_iou(self.render(pose), self.target)

    def align(self, pose: OrbitPose) -> OrbitPose:
        """Closed-form distance and offset so area and centroid match the target. Two passes,
        because changing distance moves the centroid slightly under perspective."""
        for _ in range(2):
            mask = self.render(pose)
            centroid, area = mask_centroid_and_area(mask)
            if area == 0:
                return pose
            # Projected area scales ~ 1/d^2.
            distance = max(pose.distance * math.sqrt(area / self.target_area), self.min_distance)
            pose = replace(pose, distance=distance)
            mask = self.render(pose)
            centroid, area = mask_centroid_and_area(mask)
            if area == 0:
                return pose
            shift = self.target_centroid - centroid
            pose = replace(
                pose,
                offset_x_px=pose.offset_x_px + float(shift[0]),
                offset_y_px=pose.offset_y_px + float(shift[1]),
            )
        return pose

    # Parameter vector: az, el, roll, log(distance), ox, oy, [fov]
    def to_vector(self, pose: OrbitPose) -> FloatArray:
        v = [
            pose.azimuth_deg,
            pose.elevation_deg,
            pose.roll_deg,
            math.log(pose.distance),
            pose.offset_x_px,
            pose.offset_y_px,
        ]
        if self.settings.refine_fov:
            v.append(pose.fov_deg)
        return np.array(v)

    def from_vector(self, v: FloatArray, base: OrbitPose) -> OrbitPose:
        return replace(
            base,
            azimuth_deg=float(v[0]),
            elevation_deg=float(np.clip(v[1], -MAX_ELEVATION_DEG, MAX_ELEVATION_DEG)),
            roll_deg=float(v[2]),
            distance=max(float(math.exp(v[3])), self.min_distance),
            offset_x_px=float(np.clip(v[4], -self.W, self.W)),
            offset_y_px=float(np.clip(v[5], -self.H, self.H)),
            fov_deg=float(np.clip(v[6], 10.0, 90.0)) if self.settings.refine_fov else base.fov_deg,
        )

    def loss(self, v: FloatArray, base: OrbitPose) -> float:
        cov = self.coverage(self.from_vector(v, base))
        return 1.0 - soft_iou(blur_mask(cov, self.settings.blur_sigma_px), self.target_soft)

    def nelder_mead(self, start: OrbitPose, steps: list[float], max_iter: int) -> OrbitPose:
        x0 = self.to_vector(start)
        eye = np.eye(len(x0))
        simplex = np.vstack([x0] + [x0 + eye[i] * d for i, d in enumerate(steps)])
        res = optimize.minimize(
            self.loss,
            x0,
            args=(start,),
            method="Nelder-Mead",
            options={"initial_simplex": simplex, "maxiter": max_iter, "xatol": 1e-3, "fatol": 1e-5},
        )
        return self.from_vector(res.x, start)


def _grid(center: float, half_range: float, step: float) -> FloatArray:
    if half_range <= 0:
        return np.array([center])
    n = math.floor(half_range / step)
    return np.asarray(center + step * np.arange(-n, n + 1), dtype=np.float64)


def _bounding_radius(vertices: FloatArray) -> float:
    return float(np.linalg.norm(vertices, axis=1).max())


def fit_camera(
    vertices: FloatArray,
    faces: IntArray,
    target_mask: BoolArray,
    view_label: str | None = None,
    settings: FitSettings | None = None,
    rasterizer: Rasterizer = rasterize,
) -> FitResult:
    """Fit an orbit camera so the mesh silhouette matches ``target_mask``.

    ``vertices`` must already be normalised (centred at the origin, see ``normalize.py``).
    """
    s = settings or FitSettings()
    vertices = np.asarray(vertices, dtype=np.float64)
    target_full = np.asarray(target_mask, dtype=bool)
    if not target_full.any():
        raise ValueError("target mask is empty")
    target_cov, scale = _resize_coverage(target_full, s.resolution)
    obj = _Objective(vertices, faces, target_cov, s, rasterizer)

    prior = prior_for(view_label)
    if prior.wide_search:
        azimuths = np.arange(-180.0, 180.0, s.azimuth_step_deg)
        elevations = _grid(15.0, 45.0, s.elevation_step_deg)
    else:
        azimuths = _grid(prior.azimuth_deg, s.azimuth_range_deg, s.azimuth_step_deg)
        elevations = _grid(prior.elevation_deg, s.elevation_range_deg, s.elevation_step_deg)
    elevations = elevations[np.abs(elevations) <= MAX_ELEVATION_DEG]

    # Start far enough that the whole product is in front of the camera at the given FOV.
    start_distance = 3.0 * _bounding_radius(vertices)

    # 1. Coarse search with closed-form scale and centring.
    coarse: list[Hypothesis] = []
    for az in azimuths:
        for el in elevations:
            pose = OrbitPose(float(az), float(el), start_distance, fov_deg=s.fov_deg)
            pose = obj.align(pose)
            coarse.append(Hypothesis(pose, obj.hard_iou(pose)))
    coarse.sort(key=lambda h: h.iou, reverse=True)

    # 2. Keep the top-k hypotheses that are not near-duplicates of one already kept.
    seeds: list[Hypothesis] = []
    for h in coarse:
        R = h.pose.camera(obj.W, obj.H).R
        if all(
            rotation_angle_deg(R, k.pose.camera(obj.W, obj.H).R) > 1.5 * s.azimuth_step_deg
            for k in seeds
        ):
            seeds.append(h)
        if len(seeds) == s.top_k:
            break

    # 3. Local refinement of each distinct hypothesis.
    steps = [s.azimuth_step_deg / 2, s.elevation_step_deg / 2, 5.0, 0.05, 3.0, 3.0]
    if s.refine_fov:
        steps.append(5.0)

    def refine(start: Hypothesis, scale_steps: float, max_iter: int) -> Hypothesis:
        pose = obj.nelder_mead(start.pose, [d * scale_steps for d in steps], max_iter)
        iou = obj.hard_iou(pose)
        # Never let refinement make a hypothesis worse than where it started.
        return Hypothesis(pose, iou) if iou >= start.iou else start

    refined = sorted((refine(h, 1.0, s.max_iter) for h in seeds), key=lambda h: h.iou, reverse=True)

    # 4. Polish the winner with shrinking restarts (Nelder-Mead collapses its simplex early).
    polished = refined[0]
    for i in range(s.polish_restarts):
        polished = refine(polished, 0.5 ** (i + 1), s.max_iter // 2)
    refined[0] = polished

    best = refined[0]
    best_R = best.pose.camera(obj.W, obj.H).R
    ambiguous = any(
        best.iou - h.iou <= s.ambiguity_iou_margin
        and rotation_angle_deg(best_R, h.pose.camera(obj.W, obj.H).R) > s.ambiguity_angle_deg
        for h in refined[1:]
    )
    camera = best.pose.camera(obj.W, obj.H).scaled(1.0 / scale)
    # ``scaled`` rounds the image size; pin it to the original mask exactly.
    camera = replace(camera, width=target_full.shape[1], height=target_full.shape[0])
    return FitResult(
        pose=best.pose,
        camera=camera,
        iou=best.iou,
        low_iou=best.iou < s.min_iou,
        ambiguous=ambiguous,
        alternatives=tuple(refined[1:]),
        evaluations=obj.evaluations,
    )
