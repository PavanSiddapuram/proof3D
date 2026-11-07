"""Geometry core for Proof3D: cameras, rasterisation, camera fitting, failure regions."""

from proof3d_geometry.camera import Camera, OrbitPose, orbit_camera, rotation_angle_deg
from proof3d_geometry.camera_fit import FitResult, FitSettings, fit_camera
from proof3d_geometry.mesh import MeshReport, check_mesh, normalize
from proof3d_geometry.metrics import agreement_mask, silhouette_iou
from proof3d_geometry.raster import RasterOutput, rasterize, silhouette
from proof3d_geometry.regions import FailingRegion, failing_regions

__all__ = [
    "Camera",
    "FailingRegion",
    "FitResult",
    "FitSettings",
    "MeshReport",
    "OrbitPose",
    "RasterOutput",
    "agreement_mask",
    "check_mesh",
    "failing_regions",
    "fit_camera",
    "normalize",
    "orbit_camera",
    "rasterize",
    "rotation_angle_deg",
    "silhouette",
    "silhouette_iou",
]
