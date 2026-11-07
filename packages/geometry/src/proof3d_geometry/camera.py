"""Pinhole camera and object-centric orbit poses.

Conventions (see docs/architecture.md#conventions):

* World: glTF, y-up, metres; the product is normalised to sit at the origin and faces +z.
* Camera: OpenCV, x right, y down, z forward. Pixel centres are at (i + 0.5, j + 0.5).
* Orbit: azimuth 0 deg puts the camera on +z looking at the product's front; positive azimuth
  orbits towards +x (the product's own left side). Elevation is positive above the product.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

MAX_ELEVATION_DEG = 85.0  # avoid the look-at singularity straight above the product


@dataclass(frozen=True)
class Camera:
    """World-to-camera rotation ``R``, translation ``t`` and pinhole intrinsics, in pixels."""

    R: FloatArray
    t: FloatArray
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def to_camera(self, points: FloatArray) -> FloatArray:
        return np.asarray(points, dtype=np.float64) @ self.R.T + self.t

    def project(self, points: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Return ``(uv, z)``: pixel coordinates and camera-space depth for each point."""
        pc = self.to_camera(points)
        z = pc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.fx * pc[:, 0] / z + self.cx
            v = self.fy * pc[:, 1] / z + self.cy
        return np.stack([u, v], axis=1), z

    @property
    def center(self) -> FloatArray:
        """Camera centre in world coordinates."""
        return np.asarray(-self.R.T @ self.t, dtype=np.float64)

    def scaled(self, factor: float) -> Camera:
        """Same camera for an image resized by ``factor`` (used to lift a low-res fit)."""
        return replace(
            self,
            fx=self.fx * factor,
            fy=self.fy * factor,
            cx=self.cx * factor,
            cy=self.cy * factor,
            width=round(self.width * factor),
            height=round(self.height * factor),
        )


@dataclass(frozen=True)
class OrbitPose:
    """Low-dimensional camera parameterisation searched by the fitter."""

    azimuth_deg: float
    elevation_deg: float
    distance: float
    roll_deg: float = 0.0
    fov_deg: float = 40.0
    offset_x_px: float = 0.0  # principal-point shift, absorbs off-centre product framing
    offset_y_px: float = 0.0

    def camera(self, width: int, height: int) -> Camera:
        return orbit_camera(self, width, height)


def look_at_rotation(eye: FloatArray, target: FloatArray, roll_deg: float = 0.0) -> FloatArray:
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, world_up)
    norm = np.linalg.norm(right)
    if norm < 1e-9:
        raise ValueError("look_at is singular: camera is directly above or below the target")
    right /= norm
    down = np.cross(forward, right)
    R = np.stack([right, down, forward])
    if roll_deg:
        c, s = math.cos(math.radians(roll_deg)), math.sin(math.radians(roll_deg))
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ R
    return R


def orbit_eye(azimuth_deg: float, elevation_deg: float, distance: float) -> FloatArray:
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    return distance * np.array(
        [math.sin(az) * math.cos(el), math.sin(el), math.cos(az) * math.cos(el)]
    )


def focal_from_fov(fov_deg: float, height: int) -> float:
    """Vertical field of view to focal length in pixels."""
    return 0.5 * height / math.tan(math.radians(fov_deg) / 2.0)


def orbit_camera(pose: OrbitPose, width: int, height: int) -> Camera:
    el = float(np.clip(pose.elevation_deg, -MAX_ELEVATION_DEG, MAX_ELEVATION_DEG))
    eye = orbit_eye(pose.azimuth_deg, el, pose.distance)
    R = look_at_rotation(eye, np.zeros(3), pose.roll_deg)
    f = focal_from_fov(pose.fov_deg, height)
    return Camera(
        R=R,
        t=-R @ eye,
        fx=f,
        fy=f,
        cx=width / 2.0 + pose.offset_x_px,
        cy=height / 2.0 + pose.offset_y_px,
        width=width,
        height=height,
    )


def rotation_angle_deg(R1: FloatArray, R2: FloatArray) -> float:
    """Geodesic distance between two rotations, in degrees."""
    cos = (np.trace(R1 @ R2.T) - 1.0) / 2.0
    return math.degrees(math.acos(float(np.clip(cos, -1.0, 1.0))))
