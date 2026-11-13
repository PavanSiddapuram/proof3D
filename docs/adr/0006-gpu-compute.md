# ADR 0006: GPU compute on free tiers (Modal and Kaggle)

**Status:** accepted, 2026-11-11

## Context
The GPU stages need these models:
- **TRELLIS.2, the condition A baseline.** Its model card asks for at least 24 GB of GPU memory, and it was tested on A100 and H100 cards.
- **Hunyuan3D-2mv, condition B shape.** Reported needs range from about 6 to 16 GB, depending on resolution. That figure comes from third-party guides, not a measurement here.

The PRD assumed a rented GPU. This ADR looks for free compute first.

Until now the GPU worker polled a local Postgres and wrote to local disk. Neither can be reached from a cloud GPU.

## Options
Free-tier terms below were read on 2026-11-11 from the sources listed at the end. Free tiers change without notice.

| Option | Free allowance | Memory | Verdict |
|---|---|---|---|
| **Modal, Starter plan** | $30 of credit a month, billed per second. At list prices that is about 37 h on an L4, 27 h on an A10, 15 h on an L40S or 12 h on an A100-80GB | 24 GB (L4, A10), 48 GB (L40S), 80 GB | Meets the TRELLIS.2 requirement. Serverless, so it suits a worker that drains the queue and exits |
| **Kaggle notebooks** | About 30 GPU hours a week, as users report it; Kaggle shows the actual figure in account settings. Sessions last up to 12 h | 16 GB (T4) | Enough for ingest, Hunyuan3D-2mv and scoring work. Below the TRELLIS.2 requirement |
| Google Colab, free | No published allowance; a GPU when one is available | 16 GB (T4) | Too unpredictable to plan on. Fallback only |
| Hugging Face ZeroGPU | 3.5 minutes a day | n/a | Runs only inside hosted Gradio apps. Good for looking at a photo by hand, not for the pipeline |
| Google Cloud $300 trial | Trial accounts cannot attach GPUs | n/a | Ruled out |
| Lightning AI | Reportedly a one-time grant since September 2026 | n/a | Ruled out |
| Hourly rental (RunPod and similar) | Paid | Any | The fallback if free credit runs out |

## Decision
1. **Develop on Kaggle.** Use a T4 for ingest, Hunyuan3D-2mv and the scoring code.
2. **Run TRELLIS.2 and every scored eval run on Modal.** The default GPU is an L40S. Set `PROOF3D_MODAL_GPU` to change it.
3. **Both run the same command,** `python -m proof3d_worker_gpu --drain`. It processes every queued GPU stage and exits, so paid time is never spent polling an empty queue. On Modal it is `make gpu-drain`.
4. **Shared state goes off the laptop:**
   - a hosted Postgres, for example the Neon or Supabase free tier;
   - an S3-compatible bucket, for example Cloudflare R2, set with `PROOF3D_STORAGE_BACKEND=s3`.

   The API, the CPU worker and the GPU worker all point at the same two, wherever each one runs.

The setup steps are in [docs/gpu.md](../gpu.md).

## Consequences
- **New code:**
  - `S3Storage`, an S3-compatible storage backend;
  - `drain` and `run_worker` in the worker loop;
  - `services/worker-gpu/modal_app.py`, the Modal app.

  The Modal image installs the third-party versions exported from `uv.lock`, so Modal runs the same dependency set CI tests.
- **Photos leave the laptop.** They go to the bucket, Modal and Kaggle. The dataset rule is unchanged: only products photographed by the owner. Kaggle notebooks must stay private, and all keys live in Modal or Kaggle secrets, never in the notebook or the repo.
- **Running in the cloud does not change any model's licence or territory terms.** The licence register still gates any public demo.
- **Costs are measured, not invented.**
  - Per-stage cost is measured GPU-seconds × `PROOF3D_GPU_USD_PER_HOUR`.
  - Modal also bills container start-up and model loading, which no stage records. The total spend for the report comes from Modal's usage page, alongside the per-stage sums.
  - Kaggle runs cost $0, so set `PROOF3D_GPU_USD_PER_HOUR=0` there and report Kaggle time as GPU-seconds only.
- **Kaggle and CUDA 13.** The torch version in `uv.lock` (2.15.0) is a CUDA 13 build. CUDA 13 dropped support for Pascal cards, so choose the T4 on Kaggle, not the P100. Check `torch.cuda.is_available()` before the first run, because the notebook's driver has to support CUDA 13. If it doesn't, the fix is a CUDA 12 torch build for that environment only, recorded here.
- **Compiled model extensions.** The Modal image is plain Debian with no CUDA compiler. When a stage needs custom CUDA extensions compiled, which TRELLIS.2 does, the image moves to an `nvidia/cuda` devel base. The CUDA version must match the torch build.
- **Revisit when:**
  - the $30 monthly credit runs out before the eval is done; or
  - Modal start-up time dominates short stages. The fix then is batching jobs per run, not a different provider.

## Sources
- Modal pricing: https://modal.com/pricing
- Modal GPU types: https://modal.com/docs/guide/gpu
- TRELLIS.2 model card: https://huggingface.co/microsoft/TRELLIS.2-4B
- Free-tier survey, read 2026-09-21 (Colab, Kaggle, Lightning, Studio Lab, Google Cloud trial): https://gpuperhour.com/blog/free-cloud-gpus-and-credits
- Hugging Face ZeroGPU quotas: https://huggingface.co/docs/hub/spaces-zerogpu
- Lightning AI free-tier change: https://www.usagepricing.com/blueprint/activity/lightning-ai-2026-09-21-gpu-rate-card-reprice
