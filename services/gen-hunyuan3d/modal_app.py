"""Shape generation for conditions A and B on Modal: Hunyuan3D-2 (one photo) and Hunyuan3D-2mv
(up to four views), in a container of their own (ADR 0005, ADR 0009).

    make gen-weights     # once: download the pinned weights into the proof3d-weights volume
    make gen-deploy      # deploy, so the GPU worker can call it
    make gen-smoke ARGS="--front front.png --left left.png"   # one call, writes .data/gen/*.glb

The GPU worker's generate_shape stage calls ``Hunyuan3D.single`` and ``Hunyuan3D.multi`` with
cut-out RGBA PNGs and gets GLB bytes back. The container never touches the database or the
bucket, and it has its own pinned torch and hy3dgen, so the worker's environment stays as CI
tests it.

Everything is pinned: the hy3dgen code commit, the weights revisions, and the Python packages.
Licence: Tencent Hunyuan 3D 2.0 Community License (territory excludes the EU, UK and South
Korea); see docs/licence-register.md before any public demo.

This file is run by ``modal`` (``uv run --with modal==1.6.3 modal ...``) and is not part of the
uv workspace.
"""

from __future__ import annotations

import io
import os
import time
from pathlib import Path
from typing import Any

import modal

HY3DGEN_COMMIT = "f8db63096c8282cb27354314d896feba5ba6ff8a"  # Tencent-Hunyuan/Hunyuan3D-2
SINGLE = {
    "repo": "tencent/Hunyuan3D-2",
    "revision": "9cd649ba6913f7a852e3286bad86bfa9a2d83dcf",
    "subfolder": "hunyuan3d-dit-v2-0",
}
MULTI = {
    "repo": "tencent/Hunyuan3D-2mv",
    "revision": "3a761b539b29fe4ff64714813aa9560fd66f5de0",
    "subfolder": "hunyuan3d-dit-v2-mv",
}
WEIGHTS_DIR = "/weights"  # HY3DGEN_MODELS: hy3dgen loads {dir}/{repo}/{subfolder} if it exists
GPU = os.environ.get("PROOF3D_GEN_GPU", "L4")  # 24 GB; shape generation needs ~6 GB per model
# "cpu" runs on Modal CPUs in float32: slow, but it needs no payment method on the account.
ON_CPU = GPU.lower() == "cpu"
CPU_CORES = int(os.environ.get("PROOF3D_GEN_CPU_CORES", "16"))
MAX_FACES = 200_000  # after hy3dgen's own clean-up; optimise makes the phone LODs
# Our view label -> Hunyuan3D-2mv's slot. Its left and right are mirrored against ours (left
# is azimuth +90 in proof3d_geometry.views). Measured 2026-11-18 on the synthetic tripod: with
# our left photo in its left slot, B's held-out top IoU was 0.837; in its right slot, 0.901.
MV_SLOT = {"front": "front", "left": "right", "back": "back", "right": "left"}

weights = modal.Volume.from_name("proof3d-weights", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "diffusers==0.32.2",
        "transformers==4.48.3",
        "accelerate==1.3.0",
        "huggingface_hub==0.28.1",
        "safetensors==0.5.2",
        "einops==0.8.0",
        "omegaconf==2.3.0",
        "opencv-python-headless==4.11.0.86",
        "numpy==1.26.4",
        "trimesh==4.6.1",
        "pymeshlab==2023.12.post3",
        "pygltflib==1.16.3",
        "tqdm==4.67.1",
        "pillow==11.1.0",
        # Imported by hy3dgen.shapegen but not in its requirements (marching cubes, configs).
        "scikit-image==0.24.0",
        "PyYAML==6.0.2",
    )
    # --no-deps: setup.py also pulls in gradio, rembg and onnxruntime, which shape generation
    # does not use.
    .pip_install(
        f"git+https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git@{HY3DGEN_COMMIT}",
        extra_options="--no-deps",
    )
    # Fail the build, not every container start, if --no-deps left an import out.
    .run_commands("python -c 'import hy3dgen.shapegen'")
    .env({"HY3DGEN_MODELS": WEIGHTS_DIR, "HF_HUB_DISABLE_TELEMETRY": "1"})
    # The container re-imports this file, so it must see the same settings as the machine
    # that deployed it; otherwise a CPU container would load the models onto CUDA.
    .env({"PROOF3D_GEN_GPU": GPU, "PROOF3D_GEN_CPU_CORES": str(CPU_CORES)})
)

app = modal.App("proof3d-gen-hunyuan3d", image=image)


@app.function(volumes={WEIGHTS_DIR: weights}, timeout=60 * 60)
def download_weights() -> list[str]:
    """Fetch the pinned weights into the volume. Safe to rerun: existing files are kept."""
    from huggingface_hub import snapshot_download

    got = []
    for spec in (SINGLE, MULTI):
        sub = spec["subfolder"]
        snapshot_download(
            repo_id=spec["repo"],
            revision=spec["revision"],
            allow_patterns=[f"{sub}/config.yaml", f"{sub}/model.fp16.safetensors"],
            local_dir=f"{WEIGHTS_DIR}/{spec['repo']}",
        )
        got += [str(p) for p in Path(WEIGHTS_DIR, spec["repo"], sub).iterdir()]
    weights.commit()
    return sorted(got)


@app.cls(
    gpu=None if ON_CPU else GPU,
    cpu=CPU_CORES if ON_CPU else None,
    memory=32 * 1024 if ON_CPU else None,  # both pipelines in float32 take ~14 GB
    volumes={WEIGHTS_DIR: weights},
    timeout=(90 if ON_CPU else 20) * 60,
    scaledown_window=120,
)
class Hunyuan3D:
    @modal.enter()
    def load(self) -> None:
        import torch
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

        for spec in (SINGLE, MULTI):
            if not Path(WEIGHTS_DIR, spec["repo"], spec["subfolder"], "config.yaml").exists():
                raise RuntimeError("Weights missing from the volume; run `make gen-weights` first.")
        self.pipelines = {
            name: Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                spec["repo"],
                subfolder=spec["subfolder"],
                variant="fp16",
                use_safetensors=True,
                device="cpu" if ON_CPU else "cuda",
                dtype=torch.float32 if ON_CPU else torch.float16,
            )
            for name, spec in (("single", SINGLE), ("multi", MULTI))
        }
        self.gpu = f"CPU x{CPU_CORES}" if ON_CPU else torch.cuda.get_device_name(0)

    def _run(self, name: str, image: Any, seed: int, cfg: dict[str, Any]) -> dict[str, Any]:
        import torch
        from hy3dgen.shapegen import DegenerateFaceRemover, FaceReducer, FloaterRemover

        if not ON_CPU:
            torch.cuda.synchronize()
        start = time.perf_counter()
        mesh = self.pipelines[name](
            image=image,
            num_inference_steps=cfg["steps"],
            octree_resolution=cfg["octree_resolution"],
            num_chunks=cfg["num_chunks"],
            generator=torch.manual_seed(seed),
            output_type="trimesh",
        )[0]
        mesh = FloaterRemover()(mesh)
        mesh = DegenerateFaceRemover()(mesh)
        mesh = FaceReducer()(mesh, max_facenum=MAX_FACES)
        if not ON_CPU:
            torch.cuda.synchronize()
        spec = SINGLE if name == "single" else MULTI
        return {
            "glb": mesh.export(file_type="glb"),
            "gpu_s": time.perf_counter() - start,
            "gpu": self.gpu,
            "model": f"{spec['repo']}/{spec['subfolder']}",
            "weights_revision": spec["revision"],
            "code_commit": HY3DGEN_COMMIT,
        }

    @modal.method()
    def single(self, image: bytes, seed: int, cfg: dict[str, Any]) -> dict[str, Any]:
        from PIL import Image

        return self._run("single", Image.open(io.BytesIO(image)).convert("RGBA"), seed, cfg)

    @modal.method()
    def multi(self, views: dict[str, bytes], seed: int, cfg: dict[str, Any]) -> dict[str, Any]:
        from PIL import Image

        unknown = set(views) - {"front", "left", "back", "right"}
        if unknown:
            raise ValueError(f"Hunyuan3D-2mv takes front, left, back and right, not {unknown}")
        images = {MV_SLOT[v]: Image.open(io.BytesIO(b)).convert("RGBA") for v, b in views.items()}
        return self._run("multi", images, seed, cfg)


@app.local_entrypoint()
def smoke(
    front: str,
    left: str = "",
    back: str = "",
    right: str = "",
    seed: int = 0,
    steps: int = 30,
    octree: int = 380,
    only_b: bool = False,
    out: str = ".data/gen",
) -> None:
    """One single-photo and one multi-view call on your own cut-out PNGs."""
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = {"steps": steps, "octree_resolution": octree, "num_chunks": 20_000}
    views = {
        v: Path(p).read_bytes()
        for v, p in {"front": front, "left": left, "back": back, "right": right}.items()
        if p
    }
    model = Hunyuan3D()
    runs = [] if only_b else [("A", model.single.remote(views["front"], seed, cfg))]
    if len(views) > 1:
        runs.append(("B", model.multi.remote(views, seed, cfg)))
    for condition, result in runs:
        path = out_dir / f"{condition}.glb"
        path.write_bytes(result["glb"])
        print(f"{condition}: {path} ({result['gpu_s']:.1f} s on {result['gpu']})")
