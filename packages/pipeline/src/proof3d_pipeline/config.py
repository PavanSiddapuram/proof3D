"""Experiment configs: YAML validated by Pydantic, never constants in code (standards 3, rule 2)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Cfg(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


Fraction = Annotated[float, Field(ge=0.0, le=1.0)]


class CameraFitConfig(_Cfg):
    resolution: int = Field(default=128, ge=32, le=1024, description="Long side of fit image, px")
    fov_deg: float = Field(default=40.0, gt=5.0, lt=120.0)
    refine_fov: bool = False
    azimuth_range_deg: float = Field(default=60.0, ge=0.0, le=180.0)
    azimuth_step_deg: float = Field(default=15.0, gt=0.0)
    elevation_range_deg: float = Field(default=30.0, ge=0.0, le=85.0)
    elevation_step_deg: float = Field(default=10.0, gt=0.0)
    top_k: int = Field(default=3, ge=1, description="Coarse hypotheses refined locally")
    max_iter: int = Field(default=200, ge=10)
    blur_sigma_px: float = Field(default=1.0, ge=0.0)
    supersample: int = Field(default=2, ge=1, le=4, description="Coverage supersampling factor")
    polish_restarts: int = Field(default=2, ge=0)
    min_iou: Fraction = Field(default=0.85, description="Below this the fit is flagged, not scored")
    ambiguity_iou_margin: Fraction = 0.02
    ambiguity_angle_deg: float = Field(default=25.0, gt=0.0)
    max_fit_faces: int = Field(default=20_000, ge=100, description="Decimate above this for fit")


class VerifyConfig(_Cfg):
    camera_fit: CameraFitConfig = CameraFitConfig()
    render_resolution: int = Field(default=512, ge=64)
    lpips_max: Fraction = 0.25
    dino_min: Fraction = 0.80
    flip_max: Fraction = 0.20
    region_error_threshold: Fraction = 0.35
    region_min_face_pixels: int = Field(default=4, ge=1)
    leave_one_out: bool = False


class OptimizeConfig(_Cfg):
    lod_ratios: tuple[float, ...] = (1.0, 0.35, 0.1)
    texture_max_px: int = Field(default=2048, ge=256)
    flip_views: int = Field(default=8, ge=1)
    flip_mean_max: Fraction = 0.08


class PipelineConfig(_Cfg):
    name: str = "default"
    seed: int = 0
    n_input_photos: int | None = Field(default=None, ge=1, description="Fewer-photo study (PRD 6)")
    verify: VerifyConfig = VerifyConfig()
    optimize: OptimizeConfig = OptimizeConfig()


def load_config(path: str | Path) -> PipelineConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return PipelineConfig.model_validate(data)


def config_hash(cfg: BaseModel) -> str:
    """Stable short hash of a fully-resolved config (defaults included)."""
    canonical = json.dumps(cfg.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
