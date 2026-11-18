# ADR 0009: Generation models run in their own Modal containers

**Status:** accepted, 2026-11-18

## Context
The GPU worker runs from the uv-locked workspace, the same dependency set CI tests (ADR 0006). The generation models each pin their own stack:
- **Hunyuan3D-2:** hy3dgen from git, a specific transformers range, and pymeshlab.
- **TRELLIS.2, if it comes back:** PyTorch 2.6 with CUDA 12.4, flash-attn and several custom CUDA extensions.

Installing these into the worker's environment would tie every stage to one model's pins, pull torch builds onto the dev lock, and make the worker image as fragile as the most fragile model.

The development laptop's GPU (Quadro P2200, 5 GB, Pascal) cannot run them either. The locked torch is a CUDA 13 build, which dropped Pascal.

## Decision
1. **Each model family is its own Modal app,** with a pinned image and weights in a shared Modal volume (`proof3d-weights`). The first is `services/gen-hunyuan3d/modal_app.py` (app `proof3d-gen-hunyuan3d`). It is not a workspace package.
2. **The `generate_shape` stage calls it remotely** (`modal.Cls.from_name`). It sends cut-out PNG bytes and gets GLB bytes back. The container never sees the database, the bucket or any secret; the worker stores the result.
3. **The stage records the container's GPU time** as its `gpu_s`. Cost uses `PROOF3D_GEN_GPU_USD_PER_HOUR`, the list price of `PROOF3D_GEN_GPU` (default L4). It stays unknown when the price is unset.
4. **Failures:**
   - an unreachable or undeployed app is retried, then reported;
   - a crashed call (out of memory, preemption) is retried, and nothing from the failed attempt is kept;
   - a result that is not a GLB, or is empty, fails the stage.
5. **Commands:** `make gen-weights` once, then `make gen-deploy`. `make gen-smoke` makes one A and one B call on your own cut-outs.

## Consequences
- **The worker can call generation from anywhere** Modal is logged in: the laptop, or the Modal-hosted drain.
- **Container start-up and model loading are billed by Modal but not recorded per stage.** The first call after scale-down loads two models; `scaledown_window` keeps the container warm for 2 minutes between calls.
- **The container's Python packages are pinned in its file, not in `uv.lock`.** CI does not build it. It has to be checked on Modal after any change, and `make gen-smoke` is that check.
- **Tests use a stand-in generator.** The real container is exercised only on Modal.
