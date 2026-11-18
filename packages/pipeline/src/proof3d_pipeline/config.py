"""Experiment configs: YAML validated by Pydantic, never constants in code (standards 3, rule 2)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from proof3d_pipeline.models import Condition


class _Cfg(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


Fraction = Annotated[float, Field(ge=0.0, le=1.0)]


class IngestConfig(_Cfg):
    # Background removal for photos without transparency. None accepts background-removed
    # photos only and rejects the rest, which needs no model.
    segmenter: Literal["birefnet-general-lite"] | None = "birefnet-general-lite"
    mask_threshold: Fraction = Field(default=0.5, description="Product probability cut-off")


class GenerateConfig(_Cfg):
    """Shape generation for conditions A and B (ADR 0005, ADR 0009)."""

    # A uses Hunyuan3D-2 on one photo, B uses Hunyuan3D-2mv on up to four labelled views.
    backend: Literal["hunyuan3d-2"] = "hunyuan3d-2"
    steps: int = Field(default=50, ge=5, le=100, description="Diffusion steps")
    octree_resolution: int = Field(default=380, ge=128, le=512, description="Surface grid")
    num_chunks: int = Field(default=20_000, ge=1_000, description="Decoder batch size")


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
    # Which evaluation conditions this job builds (PRD 6). C projects real photos onto B's
    # shape, and D repairs C, so each needs the one before it.
    conditions: tuple[Condition, ...] = (
        Condition.A_SINGLE_GENERATED,
        Condition.B_MULTI_GENERATED,
        Condition.C_MULTI_PROJECTED,
        Condition.D_MULTI_PROJECTED_REPAIRED,
    )
    ingest: IngestConfig = IngestConfig()
    generate: GenerateConfig = GenerateConfig()
    verify: VerifyConfig = VerifyConfig()
    optimize: OptimizeConfig = OptimizeConfig()

    @model_validator(mode="after")
    def _conditions_are_buildable(self) -> PipelineConfig:
        wanted = set(self.conditions)
        if Condition.EXTERNAL in wanted:
            raise ValueError("'external' is set by importing a model, not by the config")
        if not wanted:
            raise ValueError("conditions must name at least one of A, B, C, D")
        needs = {
            Condition.C_MULTI_PROJECTED: Condition.B_MULTI_GENERATED,
            Condition.D_MULTI_PROJECTED_REPAIRED: Condition.C_MULTI_PROJECTED,
        }
        for condition, needed in needs.items():
            if condition in wanted and needed not in wanted:
                raise ValueError(f"condition {condition} needs condition {needed}")
        return self


def load_config(path: str | Path) -> PipelineConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return PipelineConfig.model_validate(data)


def config_hash(cfg: BaseModel) -> str:
    """Stable short hash of a fully-resolved config (defaults included)."""
    canonical = json.dumps(cfg.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
