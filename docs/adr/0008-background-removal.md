# ADR 0008: Background removal with BiRefNet through ONNX Runtime

**Status:** accepted, 2026-11-16

## Context
Until now, ingest accepted only background-removed PNGs and rejected every other photo. Merchants' listing photos are mostly JPEGs on a backdrop, so Proof3D could not take the photos merchants already have. The PRD names BiRefNet as the mask model.

BiRefNet can run three ways:
- **Hugging Face `transformers` with `trust_remote_code=True`.** This downloads and executes Python from the model repository at load time, and pulls in torch, timm, kornia and einops.
- **The authors' PyTorch code.** This vendors their repository and needs torch.
- **The authors' own ONNX export through ONNX Runtime.** This is a single file with no model code to execute. It runs on CPU in the dev environment, and on a GPU through the CUDA provider.

## Decision
1. **Run BiRefNet through ONNX Runtime from the authors' ONNX export** (GitHub release `v1` of ZhengPeng7/BiRefNet). No remote code runs.
2. **Start with the small model,** `birefnet-general-lite` (Swin-v1-tiny backbone, 214 MB). The config names the model, so the config hash records it. The full model (928 MB) is added once a GPU run shows the small one is not good enough.
3. **Download once and check against a pinned SHA-256** in `PROOF3D_MODEL_DIR`. A file with the wrong hash is never loaded.
4. **Transparency wins.** A photo that is already a cut-out keeps its own mask, and the model loads only when a photo needs it.
5. **The model can be turned off.** With `ingest.segmenter: null` in the config, opaque photos are rejected with that reason.

## Consequences
- **Ordinary JPEGs work end to end.** On the synthetic tabletop demo, BiRefNet's masks gave the same verdicts as the cut-out PNGs, including catching the wrong-depth model. The demo product is synthetic and easy: one solid-coloured object on a plain backdrop. Real products, especially glass, shine, thin parts or busy backgrounds, have not been tested yet.
- **The CPU path is slow and memory-hungry.** On an 8-core laptop it takes about 17 s to load and 13–19 s per photo, with a peak of about 6 GB of RAM per 1024 px photo. That peak is with ONNX Runtime's memory arena off, which saves about 1.2 GB for about 1.5 s per photo. Running out of memory fails the stage with a clear, retryable message.
- **The GPU path is not set up yet.** It needs `onnxruntime-gpu` in the GPU image, and a cached copy of the weights on Modal, so each container does not download 214 MB.
- **Licence.** The code and weights are listed as MIT. The general model's training mix includes datasets whose own terms have not been checked (licence register). It stays "to verify" before any public demo.
- **Tests stay fast.** The default suite uses a stand-in segmenter. The real-weights test runs only with `PROOF3D_RUN_MODEL_TESTS=1`.
