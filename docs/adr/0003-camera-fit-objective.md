# ADR 0003: Camera fitting objective and CPU reference path

**Status:** accepted, 2026-11-07

## Context
Verification depends on fitting a camera to a held-out photo using only its mask (PRD 5.4). A bad fit silently corrupts every downstream metric. CI has no GPU.

## Options
1. **Gradient descent through nvdiffrast only.** Fast and accurate, but GPU-only, so CI cannot test the logic.
2. **Derivative-free search on binary IoU.** Simple, but the objective is piecewise constant, and Nelder-Mead stalled up to 12° away from the true pose.
3. **Derivative-free search on soft IoU of supersampled coverage maps, seeded by a coarse grid around the view-label prior, with a CPU rasteriser; nvdiffrast as a faster backend for the same objective.**

## Decision
Option 3. The CPU rasteriser is property-tested against a per-face reference loop and is the reference for any GPU backend. Fits report `low_iou` and `ambiguous` flags, and low-IoU fits are not scored.

## Consequences
- CI tests pose recovery end to end: under 5° on four known poses at the default 128 px.
- A CPU fit takes about 7 s on a small mesh. Generated meshes are decimated to `max_fit_faces` before fitting, because silhouettes don't need detail.
- The GPU backend must reproduce the CPU conventions (pixel centres, no culling, near-plane drop) and pass the same tests.
