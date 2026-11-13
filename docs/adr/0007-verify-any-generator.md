# ADR 0007: Verify models from any generator

**Status:** accepted, 2026-11-12

## Context
Several products already turn product photos into 3D:
- Shopify apps;
- Tripo and Rodin, which accept several images;
- Google's 360° spins;
- Ctruh's VersaAI.

Generating 3D is no longer what sets Proof3D apart. Checking each product against a photo the generator never saw is, along with saying honestly how much of the surface no photo shows. If Proof3D can only check its own models, it cannot serve as a proof layer for anyone else's generator.

## Decision
1. **Import an existing model.** `POST /products/import` accepts an optional `model`:
   - `glb`: the model file;
   - `generator` and `generator_version`: who made it;
   - `held_out`: the photos the generator never saw. At least one is required, because only these can show accuracy rather than consistency.

   The import creates an asset version with method `external`, provenance naming the generator, and photo roles set from `held_out`.
2. **Skip generation and texturing for an imported model.** These are planned skips: stage rows created as `skipped`, with `planned_skip` set. The queue passes over them, and a rerun neither resets them nor runs them.
3. **Surface evidence and next-photo suggestions** are part of verification (`proof3d_geometry.evidence`). When the verify stage is built, it will store checkable, evidenced and invented shares on every scorecard. Today the scorecard has only `unverified_surface_share`.
4. **A CPU-only check** (`proof3d_geometry.check`, `eval/scripts/check_model.py`) runs today with the parts already built: camera fits, held-out silhouette disagreements, surface evidence and suggestions. Appearance metrics are added when the GPU verify stage is built.

## Consequences
- **Checking an imported model needs no generation GPU.** Camera fitting and evidence run on CPU; only appearance scoring needs a GPU.
- **Holding a photo back costs the generator an input.** The fewer-photo study should measure that cost, because it decides whether a proof layer is worth it to a merchant with only 3–4 photos.
- **The importer declares which photos were held out, and Proof3D trusts that.** A wrong declaration turns a consistency check into a false accuracy claim. The provenance records who made the model, so the claim can be traced.
- **An imported model's front may not face +z.** Pass no view labels (or `--no-priors`), so the camera search covers the whole orbit.
- **The `planned_skip` column is new.** With `create_tables`, existing development databases need recreating. This is the trigger, already noted in settings, to move to Alembic before real data accumulates.
