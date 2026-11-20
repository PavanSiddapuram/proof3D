# ADR 0005: Shape generation models

**Status:** accepted, 2026-11-18. Proposed on 2026-11-07, pending the licence review.

## Context
Condition B needs a multi-view image-to-3D model. Condition A needs a single-photo baseline run on the same inputs.

## Candidates
- Multi-view: **Hunyuan3D-2mv** (takes several canonical views).
- Single-photo: **TRELLIS.2**, or Hunyuan3D 2.x single-view.

Using the *same* family for A and B (Hunyuan3D single vs multi-view) isolates the effect of multiple views more cleanly than comparing different model families. The plan was to prefer that if licences allowed.

## Decision
1. **A is Hunyuan3D-2** (`tencent/Hunyuan3D-2`, `hunyuan3d-dit-v2-0`) on one photo. **B is Hunyuan3D-2mv** (`tencent/Hunyuan3D-2mv`, `hunyuan3d-dit-v2-mv`) on up to four photos labelled front, left, back and right.
2. **Both see the same inputs:** the same cut-out photos, seed, steps and surface resolution. A's photo is the front one when labelled. Held-out photos are never sent.
3. **Code and weights are pinned:**
   - hy3dgen at commit `f8db630`;
   - Hunyuan3D-2 weights at revision `9cd649b`;
   - Hunyuan3D-2mv weights at revision `3a761b5`.
4. **TRELLIS.2 is deferred.** It may come back later as a second, stronger single-photo baseline. Reasons, from its repository and `pipeline.json` read on 2026-11-18:
   - it loads `briaai/RMBG-2.0` for background removal, whose weights are non-commercial and gated;
   - it loads Meta's gated DINOv3 under Meta's own licence;
   - it needs a 24 GB+ GPU, PyTorch 2.6 with CUDA 12.4, and several custom CUDA extensions, which make it the riskiest component to build.
5. **View slots come from labels given on import** (`views`) until ingest labels views itself. B needs at least two labelled views, so "multiple photos" means at least two.

## Licence
Both models are under the Tencent Hunyuan 3D 2.0 Community License (release date 21 January 2025):
- It does not apply in the EU, the UK or South Korea. The demo runs in India, inside the territory.
- A licensee with more than 1 million monthly active users must ask Tencent for a licence.
- Outputs must not be used to improve other AI models. Proof3D trains nothing.

The hy3dgen `setup.py` header names a "non-commercial" licence, but the repository's LICENSE file is the community licence. This is recorded in the licence register.

## Consequences
- **A vs B isolates the extra views,** not a change of model family.
- **Shape only for now.** Conditions A and B are meant to carry generated texture (PRD 6). That is Hunyuan3D-Paint, which needs its own CUDA rasteriser build and about 16 GB of GPU memory, and it is the next step. Until then, A and B can be compared on silhouettes, but not on appearance.
- **Results compare with the stronger single-photo models only if TRELLIS.2 is added later.** The write-up must say A is Hunyuan3D-2, not the best single-photo model available.
- **Hunyuan3D-2mv's left and right slots are mirrored against our labels** (measured 2026-11-18; hy3dgen documents them only as "front clockwise 90" and "270"). The container maps our left to its right and our right to its left. On the synthetic tripod, B's held-out top IoU went from 0.837 to 0.901, and every input photo fit better (left 0.79 to 0.88). This is one product and one seed; recheck it on the first real photos.
