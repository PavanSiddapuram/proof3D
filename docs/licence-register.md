# Licence register

Every model, weight, library and dataset Proof3D depends on, with its licence and terms. **Status "to verify" means not yet checked against the upstream licence text at the pinned version.** Nothing moves to "OK" without reading the actual licence file and noting its commit or version.

| Component | Used for | Licence (to verify upstream) | Notes and risks | Status |
|---|---|---|---|---|
| Hunyuan3D-2mv weights | Condition B shape | Tencent Hunyuan 3D community licence | Check territory exclusions (reported to exclude EU, UK, South Korea) and any user-count limits | to verify |
| TRELLIS.2 weights | Condition A baseline | Reported MIT | Confirm weights vs code licence separately | to verify |
| BiRefNet | Background masks | Reported MIT | Check the licence of the specific checkpoint's training data | to verify |
| DINOv2 | Feature similarity metric | Reported Apache-2.0 | | to verify |
| LPIPS | Perceptual metric | Reported BSD-2-Clause | The backbone weights (AlexNet/VGG) have their own terms | to verify |
| NVIDIA FLIP | Pixel metric | Reported BSD-3-Clause | | to verify |
| nvdiffrast | GPU rendering, fitting | NVIDIA Source Code License | **Reported non-commercial / research use only.** Fine for this POC; a commercial product would need a licence or a different rasteriser (the CPU path here is ours) | to verify |
| xatlas | UV unwrapping | Reported MIT | | to verify |
| meshoptimizer | Simplification, LODs, compression | Reported MIT | | to verify |
| glTF Transform | Packaging, KTX2 | Reported MIT | KTX-Software (toktx) is Apache-2.0 | to verify |
| trimesh, numpy, scipy | Geometry | MIT / BSD | | to verify |
| Product photo dataset | Evaluation | Own photographs | Only products photographed by the owner, in the owner's demo store | OK |

## Compute and storage services

Terms of service, not licences, but they govern where photos and outputs go (ADR 0006).

| Service | Used for | Check before use | Status |
|---|---|---|---|
| Modal | GPU runs: TRELLIS.2, scored eval | Data handling and retention for uploaded inputs; free-plan terms | to verify |
| Kaggle notebooks | GPU development runs | Notebook kept private; terms on uploaded data and outputs | to verify |
| Hosted Postgres (Neon or Supabase) | Shared job database | Free-tier limits; region | to verify |
| S3-compatible bucket (Cloudflare R2 or similar) | Photos, meshes, renders | Bucket private; key scoped to one bucket | to verify |
