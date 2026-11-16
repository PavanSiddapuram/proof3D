"""Synthetic products with known geometry, for tests and the end-to-end demo.

Never report numbers from these as results: they show the pipeline works, not how well it does
on real products (standards: only products you photographed yourself count).
"""

from __future__ import annotations

import numpy as np
import trimesh
from numpy.typing import NDArray

from proof3d_geometry.camera import OrbitPose, orbit_camera
from proof3d_geometry.mesh import normalize
from proof3d_geometry.raster import BoolArray, rasterize


def tripod_mesh(depth_arm: float = 0.35) -> trimesh.Trimesh:
    """A base block with three arms of different lengths, so every viewpoint gives a distinct
    silhouette and pose recovery is well-posed.

    ``depth_arm`` is the length of the arm along +z, which points at the front camera. A front
    photo cannot show its length, so a model that gets it wrong still matches the front view;
    only side or top photos expose it.
    """
    parts = [
        trimesh.creation.box(extents=[0.30, 0.30, 0.30]),
        trimesh.creation.box(extents=[1.00, 0.12, 0.12]),
        trimesh.creation.box(extents=[0.12, 0.60, 0.12]),
        trimesh.creation.box(extents=[0.12, 0.12, depth_arm]),
    ]
    offsets = [[0, 0, 0], [0.5, 0, 0], [0, 0.3, 0], [0, 0, depth_arm / 2]]
    for part, off in zip(parts, offsets, strict=True):
        part.apply_translation(off)
    mesh = trimesh.util.concatenate(parts)
    assert isinstance(mesh, trimesh.Trimesh)
    v, _, _ = normalize(mesh.vertices)
    return trimesh.Trimesh(v, mesh.faces, process=False)


def photo_mask(
    mesh: trimesh.Trimesh,
    azimuth_deg: float,
    elevation_deg: float,
    width: int = 160,
    height: int = 120,
    distance: float = 2.8,
) -> BoolArray:
    """The product mask a photo from this orbit position would have."""
    camera = orbit_camera(OrbitPose(azimuth_deg, elevation_deg, distance), width, height)
    return rasterize(mesh.vertices, mesh.faces, camera).mask


def opaque_photo(
    mesh: trimesh.Trimesh,
    azimuth_deg: float,
    elevation_deg: float,
    width: int = 480,
    height: int = 360,
    distance: float = 2.8,
    seed: int = 0,
) -> tuple[NDArray[np.uint8], BoolArray]:
    """An RGB photo with no transparency, and its true product mask.

    The product is a flat-shaded terracotta object; the background is a lit, tiled tabletop
    with mild noise, so a background remover has real work to do.
    """
    camera = orbit_camera(OrbitPose(azimuth_deg, elevation_deg, distance), width, height)
    face_id = rasterize(mesh.vertices, mesh.faces, camera).face_id
    mask = face_id >= 0

    light = np.array([0.4, 0.8, 0.6])
    light /= np.linalg.norm(light)
    shade = 0.35 + 0.65 * np.clip(np.asarray(mesh.face_normals) @ light, 0.0, None)
    product = shade[np.maximum(face_id, 0), None] * np.array([196.0, 104.0, 72.0])

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    tiles = ((xx // 40 + yy // 40) % 2) * 18.0
    glow = 1.0 - 0.35 * np.hypot(xx / width - 0.5, yy / height - 0.3)
    noise = np.random.default_rng(seed).normal(0.0, 4.0, (height, width, 1))
    background = (np.array([150.0, 160.0, 175.0]) + tiles[..., None]) * glow[..., None] + noise
    rgb = np.where(mask[..., None], product, background)
    return np.clip(rgb, 0, 255).astype(np.uint8), mask
