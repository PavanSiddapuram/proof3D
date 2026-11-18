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
MAX_FACES = 200_000  # after hy3dgen's own clean-up; optimise makes the phone LODs

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
    )
    # --no-deps: setup.py also pulls in gradio, rembg and onnxruntime, which shape generation
    # does not use.
    .pip_install(
        f"git+https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git@{HY3DGEN_COMMIT}",
        extra_options="--no-deps",
    )
    .env({"HY3DGEN_MODELS": WEIGHTS_DIR, "HF_HUB_DISABLE_TELEMETRY": "1"})
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


@app.cls(gpu=GPU, volumes={WEIGHTS_DIR: weights}, timeout=20 * 60, scaledown_window=120)
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
                device="cuda",
            )
            for name, spec in (("single", SINGLE), ("multi", MULTI))
        }
        self.gpu = torch.cuda.get_device_name(0)

    def _run(self, name: str, image: Any, seed: int, cfg: dict[str, Any]) -> dict[str, Any]:
        import torch
        from hy3dgen.shapegen import DegenerateFaceRemover, FaceReducer, FloaterRemover

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
        images = {v: Image.open(io.BytesIO(b)).convert("RGBA") for v, b in views.items()}
        return self._run("multi", images, seed, cfg)


@app.local_entrypoint()
def smoke(
    front: str, left: str = "", back: str = "", right: str = "", seed: int = 0, steps: int = 30
) -> None:
    """One single-photo and one multi-view call on your own cut-out PNGs."""
    out = Path(".data/gen")
    out.mkdir(parents=True, exist_ok=True)
    cfg = {"steps": steps, "octree_resolution": 380, "num_chunks": 20_000}
    views = {
        v: Path(p).read_bytes()
        for v, p in {"front": front, "left": left, "back": back, "right": right}.items()
        if p
    }
    model = Hunyuan3D()
    runs = [("A", model.single.remote(views["front"], seed, cfg))]
    if len(views) > 1:
        runs.append(("B", model.multi.remote(views, seed, cfg)))
    for condition, result in runs:
        path = out / f"{condition}.glb"
        path.write_bytes(result["glb"])
        print(f"{condition}: {path} ({result['gpu_s']:.1f} s on {result['gpu']})")
