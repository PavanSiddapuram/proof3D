"""Geometry core for Proof3D: cameras, rasterisation, camera fitting, failure regions."""

from proof3d_geometry.camera import Camera, OrbitPose, orbit_camera, rotation_angle_deg
from proof3d_geometry.camera_fit import FitResult, FitSettings, fit_camera
from proof3d_geometry.check import ModelCheck, PhotoCheck, PhotoInput, check_model
from proof3d_geometry.evidence import (
    EvidenceSettings,
    SurfaceEvidence,
    ViewSuggestion,
    suggest_views,
    surface_evidence,
    visible_faces,
)
from proof3d_geometry.mesh import MeshReport, check_mesh, cluster_decimate, normalize
from proof3d_geometry.metrics import agreement_mask, silhouette_iou
from proof3d_geometry.raster import RasterOutput, rasterize, silhouette
from proof3d_geometry.regions import FailingRegion, failing_regions

__all__ = [
    "Camera",
    "EvidenceSettings",
    "FailingRegion",
    "FitResult",
    "FitSettings",
    "MeshReport",
    "ModelCheck",
    "OrbitPose",
    "PhotoCheck",
    "PhotoInput",
    "RasterOutput",
    "SurfaceEvidence",
    "ViewSuggestion",
    "agreement_mask",
    "check_mesh",
    "check_model",
    "cluster_decimate",
    "failing_regions",
    "fit_camera",
    "normalize",
    "orbit_camera",
    "rasterize",
    "rotation_angle_deg",
    "silhouette",
    "silhouette_iou",
    "suggest_views",
    "surface_evidence",
    "visible_faces",
]
