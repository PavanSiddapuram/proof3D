# ADR 0004: Texture from projected listing photos

**Status:** proposed, 2026-11-07. Decide in week 3 with measurements.

## Context
Generated textures are guesses; the listing photos are ground truth for every surface they see (PRD 5.3).

## Options
1. Keep the generator's texture (conditions A and B).
2. Back-project input photos onto xatlas UVs, weighted by view angle and visibility, with seam blending. Fill unseen texels from the generated texture and mark them unverified.
3. Optimise the texture through a differentiable renderer against all input photos.

## Leaning
Option 2 for condition C: it is deterministic, explainable, and makes "unverified surface" exact. Option 3 is the stretch if seams dominate the error.

## To measure before accepting
Masked LPIPS and DINOv2 similarity for C vs B on the held-out photo, plus the unverified-surface share.
