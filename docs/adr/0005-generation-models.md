# ADR 0005: Shape generation models

**Status:** proposed, 2026-11-07. Blocked on the licence review in `docs/licence-register.md`.

## Context
Condition B needs a multi-view image-to-3D model. Condition A needs a single-photo baseline run on the same inputs.

## Candidates
- Multi-view: **Hunyuan3D-2mv** (takes several canonical views).
- Single-photo: **TRELLIS.2**, or Hunyuan3D 2.x single-view.

## Open questions
- Licence and territory terms for each (see the register). Some Hunyuan3D licences exclude territories, and that must be resolved before any public demo.
- How to map listing photos with imperfect angles onto the model's fixed view slots: nearest photo per slot, chosen from the fitted or labelled azimuth.
- Using the *same* family for A and B (Hunyuan3D single vs multi-view) isolates the effect of multiple views more cleanly than comparing different model families. Prefer that if licences allow.
