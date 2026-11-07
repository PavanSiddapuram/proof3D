"""CPU rasteriser: silhouette, depth and face-ID buffers.

This is the backend for CI and for camera fitting on decimated meshes. The GPU worker uses
nvdiffrast behind the same ``Rasterizer`` signature; ``tests/test_raster.py`` pins the
conventions every backend must share:

* pixel centres at (i + 0.5, j + 0.5); a pixel is covered if its centre is inside or on an edge;
* no back-face culling (generated meshes have unreliable winding; silhouettes must not care);
* nearest depth wins, depth interpolated perspective-correctly;
* triangles with any vertex behind the near plane are dropped, not clipped.

``rasterize`` is vectorised over faces. ``rasterize_reference`` is the plain per-face loop it is
property-tested against.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from proof3d_geometry.camera import Camera, FloatArray

IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

NEAR = 1e-4
_CHUNK_PIXELS = 2_000_000  # padded candidate pixels evaluated per vectorised batch


@dataclass(frozen=True)
class RasterOutput:
    face_id: NDArray[np.int32]  # (H, W), -1 where no face
    depth: FloatArray  # (H, W), camera-space z, +inf where no face

    @property
    def mask(self) -> BoolArray:
        return np.asarray(self.face_id >= 0)


class Rasterizer(Protocol):
    def __call__(self, vertices: FloatArray, faces: IntArray, camera: Camera) -> RasterOutput: ...


@dataclass(frozen=True)
class _Setup:
    tri_uv: FloatArray  # (F, 3, 2)
    tri_z: FloatArray  # (F, 3)
    lo: IntArray  # (F, 2) inclusive pixel bbox, x then y
    hi: IntArray
    area: FloatArray  # (F,) signed doubled area in pixels
    faces: IntArray  # indices of faces worth rasterising


def _setup(vertices: FloatArray, faces: IntArray, camera: Camera) -> _Setup:
    H, W = camera.height, camera.width
    uv, z = camera.project(np.asarray(vertices, dtype=np.float64))
    faces = np.asarray(faces, dtype=np.int64)
    tri_uv = uv[faces]
    tri_z = z[faces]
    ok = np.all(tri_z > NEAR, axis=1) & np.all(np.isfinite(tri_uv), axis=(1, 2))
    safe = np.where(ok[:, None, None], tri_uv, 0.0)
    lo = np.maximum(np.floor(safe.min(axis=1) - 0.5), 0).astype(np.int64)
    hi = np.minimum(np.ceil(safe.max(axis=1) - 0.5), [W - 1, H - 1]).astype(np.int64)
    a, b, c = safe[:, 0], safe[:, 1], safe[:, 2]
    area = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    ok &= np.all(hi >= lo, axis=1) & (np.abs(area) > 1e-12)
    return _Setup(safe, tri_z, lo, hi, area, np.flatnonzero(ok))


def _barycentric(
    tri: FloatArray, area: FloatArray, X: FloatArray, Y: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Barycentric weights; ``tri`` is (..., 3, 2) broadcastable against X and Y."""
    a, b, c = tri[..., 0, :], tri[..., 1, :], tri[..., 2, :]
    w0 = ((b[..., 0] - X) * (c[..., 1] - Y) - (b[..., 1] - Y) * (c[..., 0] - X)) / area
    w1 = ((c[..., 0] - X) * (a[..., 1] - Y) - (c[..., 1] - Y) * (a[..., 0] - X)) / area
    return w0, w1, 1.0 - w0 - w1


def _batches(s: _Setup) -> list[tuple[IntArray, int, int]]:
    """Group faces into batches padded to a common bbox size.

    Faces are bucketed by (log2 width, log2 height) of their pixel bbox, so padding is at most
    4x the real footprint, then each bucket is split to respect the memory budget. Planning is
    O(buckets) in Python, not O(faces): dense generated meshes have tens of thousands of faces.
    """
    size = s.hi[s.faces] - s.lo[s.faces] + 1
    key = np.floor(np.log2(size)).astype(np.int64)
    out: list[tuple[IntArray, int, int]] = []
    uniq, inverse = np.unique(key, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    for b in range(len(uniq)):
        members = s.faces[inverse == b]
        bw, bh = (size[inverse == b]).max(axis=0)
        per = max(1, _CHUNK_PIXELS // int(bw * bh))
        for i in range(0, members.size, per):
            out.append((members[i : i + per], int(bw), int(bh)))
    return out


def rasterize(vertices: FloatArray, faces: IntArray, camera: Camera) -> RasterOutput:
    """Vectorised z-buffer rasteriser (see module docstring for conventions)."""
    H, W = camera.height, camera.width
    s = _setup(vertices, faces, camera)
    depth_flat = np.full(H * W, np.inf)
    id_flat = np.full(H * W, -1, dtype=np.int32)
    if s.faces.size == 0:
        return RasterOutput(id_flat.reshape(H, W), depth_flat.reshape(H, W))

    pix_parts: list[IntArray] = []
    z_parts: list[FloatArray] = []
    f_parts: list[IntArray] = []
    for batch, bw, bh in _batches(s):
        ox = s.lo[batch, 0][:, None, None] + np.arange(bw)[None, None, :]
        oy = s.lo[batch, 1][:, None, None] + np.arange(bh)[None, :, None]
        X, Y = ox + 0.5, oy + 0.5
        tri = s.tri_uv[batch][:, None, None]  # (B, 1, 1, 3, 2)
        w0, w1, w2 = _barycentric(tri, s.area[batch][:, None, None], X, Y)
        inside = (
            (w0 >= 0)
            & (w1 >= 0)
            & (w2 >= 0)
            & (ox <= s.hi[batch, 0][:, None, None])
            & (oy <= s.hi[batch, 1][:, None, None])
        )
        if not inside.any():
            continue
        zs = s.tri_z[batch][:, None, None]
        zf = 1.0 / (w0 / zs[..., 0] + w1 / zs[..., 1] + w2 / zs[..., 2])
        bi, yi, xi = np.nonzero(inside)
        pix_parts.append((oy[bi, yi, 0] * W + ox[bi, 0, xi]).astype(np.int64))
        z_parts.append(zf[bi, yi, xi])
        f_parts.append(batch[bi])

    if pix_parts:
        pix = np.concatenate(pix_parts)
        zz = np.concatenate(z_parts)
        ff = np.concatenate(f_parts)
        # Nearest fragment per pixel; ties go to the lower face index, as in the reference.
        sel = np.lexsort((ff, zz, pix))
        pix, zz, ff = pix[sel], zz[sel], ff[sel]
        first = np.ones(pix.size, dtype=bool)
        first[1:] = pix[1:] != pix[:-1]
        depth_flat[pix[first]] = zz[first]
        id_flat[pix[first]] = ff[first]
    return RasterOutput(id_flat.reshape(H, W), depth_flat.reshape(H, W))


def rasterize_reference(vertices: FloatArray, faces: IntArray, camera: Camera) -> RasterOutput:
    """Per-face loop. Slow; exists to test ``rasterize`` and any future backend against."""
    H, W = camera.height, camera.width
    s = _setup(vertices, faces, camera)
    face_id = np.full((H, W), -1, dtype=np.int32)
    depth = np.full((H, W), np.inf)
    for f in s.faces:
        (x0, y0), (x1, y1) = s.lo[f], s.hi[f]
        X, Y = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        w0, w1, w2 = _barycentric(s.tri_uv[f], s.area[f], X, Y)
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        za, zb, zc = s.tri_z[f]
        zf = 1.0 / (w0 / za + w1 / zb + w2 / zc)
        region_d = depth[y0 : y1 + 1, x0 : x1 + 1]
        closer = inside & (zf < region_d)
        region_d[closer] = zf[closer]
        face_id[y0 : y1 + 1, x0 : x1 + 1][closer] = f
    return RasterOutput(face_id=face_id, depth=depth)


def silhouette(vertices: FloatArray, faces: IntArray, camera: Camera) -> BoolArray:
    return rasterize(vertices, faces, camera).mask
