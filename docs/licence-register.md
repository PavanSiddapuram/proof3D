# Licence register

Every model, weight, library and dataset Proof3D depends on, with its licence and terms. **Status "to verify" means not yet checked against the upstream licence text at the pinned version.** Nothing moves to "OK" without reading the actual licence file and noting its commit or version.

| Component | Used for | Licence (to verify upstream) | Notes and risks | Status |
|---|---|---|---|---|
| Hunyuan3D-2 and Hunyuan3D-2mv weights (`tencent/Hunyuan3D-2` @ `9cd649b`, `tencent/Hunyuan3D-2mv` @ `3a761b5`) and hy3dgen code (@ `f8db630`) | Conditions A and B shape (ADR 0005) | Tencent Hunyuan 3D 2.0 Community License, released 2025-01-21; LICENSE files read 2026-11-18 | Territory excludes the EU, UK and South Korea; the demo runs in India. Over 1M monthly active users needs Tencent's licence. Outputs must not improve other AI models. hy3dgen's `setup.py` header says "non-commercial", but its LICENSE file is the community licence; ask Tencent before any commercial use | OK for the India demo; commercial use to verify |
| TRELLIS.2 weights (`microsoft/TRELLIS.2-4B` @ `af44b45`) | Deferred (ADR 0005) | MIT per the model card | Its pipeline loads `briaai/RMBG-2.0` (non-commercial, gated) and Meta's DINOv3 (gated, Meta licence); it depends on nvdiffrast and nvdiffrec | not used |
| BiRefNet (`birefnet-general-lite`: `BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx`, GitHub release v1, SHA-256 `56000243…3333`) | Background masks (ADR 0008) | MIT per the repository and model card; LICENSE text not yet read at a pinned commit | The general model's training mix (DIS5K, DUTS, HRSOD, UHRSD, HRS10K, P3M-10k, human sets) includes datasets with their own terms, some possibly research-only. Check them before any public demo | to verify |
| DINOv2 | Feature similarity metric | Reported Apache-2.0 | | to verify |
| LPIPS | Perceptual metric | Reported BSD-2-Clause | The backbone weights (AlexNet/VGG) have their own terms | to verify |
| NVIDIA FLIP | Pixel metric | Reported BSD-3-Clause | | to verify |
| nvdiffrast | GPU rendering, fitting | NVIDIA Source Code License | **Reported non-commercial / research use only.** Fine for this POC; a commercial product would need a licence or a different rasteriser (the CPU path here is ours) | to verify |
| xatlas | UV unwrapping | Reported MIT | | to verify |
| meshoptimizer | Simplification, LODs, compression | Reported MIT | | to verify |
| glTF Transform | Packaging, KTX2 | Reported MIT | KTX-Software (toktx) is Apache-2.0 | to verify |
| trimesh, numpy, scipy | Geometry | MIT / BSD | | to verify |
| ONNX Runtime | Runs BiRefNet | Reported MIT | | to verify |
| Product photo dataset | Evaluation | Own photographs | Only products photographed by the owner, in the owner's demo store | OK |

## Compute and storage services

Terms of service, not licences, but they govern where photos and outputs go (ADR 0006).

| Service | Used for | Check before use | Status |
|---|---|---|---|
| Modal | GPU runs: Hunyuan3D generation containers, scored eval | Data handling and retention for uploaded inputs; free-plan terms | to verify |
| Kaggle notebooks | GPU development runs | Notebook kept private; terms on uploaded data and outputs | to verify |
| Hosted Postgres (Neon or Supabase) | Shared job database | Free-tier limits; region | to verify |
| S3-compatible bucket (Cloudflare R2 or similar) | Photos, meshes, renders | Bucket private; key scoped to one bucket | to verify |
