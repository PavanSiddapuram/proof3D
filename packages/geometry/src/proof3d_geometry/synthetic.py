"""Synthetic products with known geometry, for tests and the end-to-end demo.

Never report numbers from these as results: they show the pipeline works, not how well it does
on real products (standards: only products you photographed yourself count).
"""

from __future__ import annotations

import trimesh

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
