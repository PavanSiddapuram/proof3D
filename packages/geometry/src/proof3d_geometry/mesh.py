"""Mesh normalisation (PRD 5.2) and checks that feed repair rung 1 (PRD 5.5)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from proof3d_geometry.camera import FloatArray


def normalize(vertices: FloatArray) -> tuple[FloatArray, FloatArray, float]:
    """Centre the bounding box at the origin and scale its longest side to 1.

    Returns ``(normalised_vertices, centre, scale)`` so the transform can be undone.
    Orientation is left to the generator's canonical frame; both A and B meshes come out of
    image-to-3D models that put the input front view on +z.
    """
    v = np.asarray(vertices, dtype=np.float64)
    lo, hi = v.min(axis=0), v.max(axis=0)
    centre = (lo + hi) / 2.0
    extent = float((hi - lo).max())
    if extent <= 0:
        raise ValueError("mesh has zero extent")
    scale = 1.0 / extent
    return (v - centre) * scale, centre, scale


def cluster_decimate(
    vertices: FloatArray, faces: NDArray[np.int64], max_faces: int
) -> tuple[FloatArray, NDArray[np.int64]]:
    """Shrink a mesh to at most ``max_faces`` faces by snapping vertices to a grid.

    Vertices in the same cell merge at their mean; faces that collapse are dropped. The grid
    starts at 1/512 of the mesh size and halves until the face count fits, so the result keeps
    the silhouette to within about one cell. That is enough for camera fitting and surface
    evidence, which work at a few hundred pixels. It is not for display: it ignores normals,
    UVs and topology.
    """
    if max_faces < 1:
        raise ValueError("max_faces must be at least 1")
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if len(f) <= max_faces:
        return v, f
    lo = v.min(axis=0)
    extent = float((v.max(axis=0) - lo).max()) or 1.0
    grid = 512
    while grid >= 2:
        cells = np.floor((v - lo) / extent * (grid - 1e-9)).astype(np.int64)
        _, cluster, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
        cluster = cluster.ravel()
        nf = cluster[f]
        keep = (nf[:, 0] != nf[:, 1]) & (nf[:, 1] != nf[:, 2]) & (nf[:, 0] != nf[:, 2])
        nf = np.unique(np.sort(nf[keep], axis=1), axis=0) if keep.any() else nf[keep]
        if 0 < len(nf) <= max_faces:
            centres = np.zeros((len(counts), 3))
            np.add.at(centres, cluster, v)
            return centres / counts[:, None], nf
        grid //= 2
    raise ValueError(f"cannot decimate below {max_faces} faces")


@dataclass(frozen=True)
class MeshReport:
    n_vertices: int
    n_faces: int
    watertight: bool
    winding_consistent: bool
    degenerate_faces: int
    duplicate_faces: int
    boundary_edges: int
    non_manifold_edges: int
    components: int

    @property
    def needs_geometry_repair(self) -> bool:
        return (
            self.degenerate_faces > 0
            or self.duplicate_faces > 0
            or self.non_manifold_edges > 0
            or not self.winding_consistent
        )


def _count_components(faces: NDArray[np.int64]) -> int:
    """Edge-connected face components (same rule as failing-region grouping)."""
    from proof3d_geometry.regions import face_adjacency

    n = len(faces)
    if n == 0:
        return 0
    pairs = face_adjacency(faces)
    graph = sparse.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    return int(connected_components(graph, directed=False)[0])


def check_mesh(mesh: trimesh.Trimesh) -> MeshReport:
    faces = np.asarray(mesh.faces)
    areas = np.asarray(mesh.area_faces)
    sorted_faces = np.sort(faces, axis=1)
    _, counts = np.unique(sorted_faces, axis=0, return_counts=True)

    edges = np.sort(np.asarray(mesh.edges), axis=1)
    _, edge_counts = np.unique(edges, axis=0, return_counts=True)

    return MeshReport(
        n_vertices=len(mesh.vertices),
        n_faces=len(faces),
        watertight=bool(mesh.is_watertight),
        winding_consistent=bool(mesh.is_winding_consistent),
        degenerate_faces=int(np.count_nonzero(areas <= 1e-12)),
        duplicate_faces=int((counts - 1).sum()),
        boundary_edges=int(np.count_nonzero(edge_counts == 1)),
        non_manifold_edges=int(np.count_nonzero(edge_counts > 2)),
        components=_count_components(faces),
    )
