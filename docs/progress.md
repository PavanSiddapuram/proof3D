# Progress

What has been built, what it showed, and what is next. It is updated with every change set. The [build-status table](architecture.md#build-status) says which parts of the pipeline exist. This page records the order they were built in, the numbers they produced, and what is still open.

All numbers here come from **synthetic** products (`proof3d_geometry.synthetic`). They show the pipeline works; they are not results about any generator or real product. Real results need the owner's own product photos (standards: only products you photographed yourself).

## Done

| Date | Commit | What | Evidence |
|---|---|---|---|
| 2026-11-07 | `65e6bbd` | Monorepo setup: geometry core (cameras, CPU rasteriser, camera fitting, failure regions, mesh checks), typed pipeline contracts and configs, API with a Postgres-backed queue | Unit, property and API tests; CI |
| 2026-11-13 | `9b2fa78` | GPU worker on free tiers: Modal app, Kaggle steps, S3-compatible storage, `--drain` ([ADR 0006](adr/0006-gpu-compute.md)) | Modal image builds locally; S3 storage tested against a fake client |
| 2026-11-13 | `4a41928` | Verify a model from any generator: model import, surface evidence, next-photo suggestions, CPU check CLI ([ADR 0007](adr/0007-verify-any-generator.md)) | Wrong-depth model: held-out IoU 0.52 and 0.77, flagged; correct model ≥ 0.98 |
| 2026-11-13 | `2c74e64` | Example database URL uses 127.0.0.1 (on Windows, `localhost` hangs against Docker) | — |
| 2026-11-13 | `776f5f4` | End to end through the API, queue and Postgres: `POST /uploads`, Alembic migrations, ingest and verify in their CPU forms, scorecard verdict and evidence, demo script; stage memory metering made ~3× cheaper | 126 tests, 98% coverage; demo on Postgres 16.4: correct model `silhouette_agrees` (held-out 0.963, 0.994), wrong model `disagrees` (0.552, 0.674) |
| 2026-11-16 | *uncommitted* | Background removal: BiRefNet (small model) through ONNX Runtime, pinned by SHA-256, used only for photos without transparency ([ADR 0008](adr/0008-background-removal.md)) | Real model on synthetic tabletop JPEGs: mask IoU 0.990–0.999. JPEG demo on Postgres: correct model `silhouette_agrees` (0.941, 0.986), wrong model `disagrees` (0.547, 0.663). CPU: ~17 s load, 13–19 s and ~6 GB RAM per photo |

## Next

In order. Each line names what "done" means.

1. **Try real photos.** Run the JPEG path on 2–3 of the owner's own products, including one hard case (glass, shine, or thin parts). Done when the masks and verdicts are recorded here, good or bad.
2. **Background removal on the GPU.** Add `onnxruntime-gpu` to the GPU image and cache the weights on a Modal volume. Done when one Modal drain masks a product and records GPU time and cost.
3. **View labels in ingest.** Done when the held-out photo is chosen as the angle the inputs miss, and camera fitting uses the label to narrow its search.
4. **Shape generation:** TRELLIS.2 (A) and Hunyuan3D-2mv (B), after their licences are checked. Week 1 plan, now late.
5. **Appearance scoring in verify:** render, then LPIPS, DINOv2, FLIP and edges, plus the nvdiffrast backend. Week 2.
6. **Texture projection (C) and repair (D).** Week 3.
7. **Optimise, phone harness, Shopify publish, web app and viewer.** Week 4.

## Open

- **Licences to verify before any public demo:** BiRefNet's training data, Hunyuan3D-2mv's territory terms, TRELLIS.2, nvdiffrast (reportedly non-commercial). See the [licence register](licence-register.md).
- **Not yet run on real infrastructure:** Modal (no account run yet), a real S3 bucket, Kaggle.
- **Ingest still lacks** fetching photos from URLs; photos must be uploaded first.
- **The schedule:** the plan put shape generation in week 1. It has not started, because verifying other generators' models came first ([ADR 0007](adr/0007-verify-any-generator.md)).
