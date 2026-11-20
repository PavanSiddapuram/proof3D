# Running the GPU worker on Modal or Kaggle

This guide puts the GPU worker on free compute. The reasons are in [ADR 0006](adr/0006-gpu-compute.md).

The worker needs only two things: a database and a bucket it can reach. Both run the same command, `python -m proof3d_worker_gpu --drain`. The worker processes every queued GPU stage, prints how many it processed, and exits.

> **Status, 2026-11-11.** No GPU stage is built yet, so a drain today marks `ingest` as "not built yet" and stops. These steps are for checking the plumbing before the models arrive. The Modal app loads and its image definition builds locally. It has not yet run on a Modal account.

## 1. Shared database and bucket (once)

Everything that touches a job must see the same data: the API on your laptop, the CPU worker, and the GPU worker on Modal or Kaggle.

1. **Postgres:** create a free hosted database, for example on Neon or Supabase. The URL looks like `postgresql+psycopg://USER:PASSWORD@HOST/DB?sslmode=require`.
2. **Bucket:** create an S3-compatible bucket, for example on Cloudflare R2. Make an access key scoped to that bucket only.
   - For R2, the endpoint is `https://<account-id>.r2.cloudflarestorage.com` and the region is `auto`.
3. **Your `.env`:** put the values in your local `.env`. Never commit it; the gitleaks scan in CI fails on secrets.

```bash
PROOF3D_DATABASE_URL=postgresql+psycopg://...?sslmode=require
PROOF3D_STORAGE_BACKEND=s3
PROOF3D_S3_BUCKET=proof3d
PROOF3D_S3_ENDPOINT=https://<account-id>.r2.cloudflarestorage.com
PROOF3D_S3_REGION=auto
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

Then create the schema once with `make migrate` (workers and the API never create tables), start the API with `make api`, and import a product, so there is a queued GPU stage to process. `make demo ARGS="--no-worker"` uploads and imports a synthetic one.

Ingest and verify already work in their CPU form: masks come from photo transparency or from BiRefNet ([ADR 0008](adr/0008-background-removal.md)), and only silhouettes are scored. They need no GPU, so `make worker-gpu-local` runs them on your laptop. BiRefNet on a CPU needs about 6 GB of free RAM and 15 s per photo. On Modal it still runs on the CPU until `onnxruntime-gpu` is added to the image. Modal and Kaggle are needed once generation, rendering and appearance scoring are built.

## 2. Modal: the GPU worker and scored eval runs

1. **Sign up and log in.** Sign up at modal.com; the free plan gives $30 of credit a month. Then run:

   ```bash
   uv run --with modal==1.6.3 modal setup
   ```

2. **Store the configuration as a Modal secret** named `proof3d`. Put only what the GPU worker needs in a separate file, `.env.modal`, which git already ignores:
   - the seven database, bucket and AWS lines from step 1;
   - `PROOF3D_GPU_USD_PER_HOUR`, set to Modal's list price for the GPU you chose, so stage costs are real.

   Leave the Shopify token out; the GPU worker never publishes. Load the secret from the file rather than typing keys on the command line, where they would end up in your shell history:

   ```bash
   uv run --with modal==1.6.3 modal secret create proof3d --from-dotenv .env.modal
   ```

   The app refuses to start if the secret points at a local database or at local-disk storage.

3. **Process the queue:**

   ```bash
   make gpu-drain                                   # default GPU: L40S, 48 GB
   PROOF3D_MODAL_GPU=L4 make gpu-drain              # 24 GB, cheaper per hour
   make gpu-drain ARGS="--max-stages 1"             # one stage, to test
   ```

   - The first run builds the image, which installs torch and takes a few minutes. Later runs reuse it.
   - The image installs the exact third-party versions from `uv.lock`, exported on your machine at build time, plus the workspace packages.

4. **Check spend.** Compare Modal's usage page with the per-stage costs in the job. Modal also bills container start-up and model loading, which no stage records.

## 3. Shape generation on Modal (ADR 0009)

The generation models run in their own container, which `generate_shape` calls. Set it up once, after `modal setup`:

```bash
make gen-weights    # download the pinned Hunyuan3D-2 and -2mv weights into the proof3d-weights volume
make gen-deploy     # deploy the container (app proof3d-gen-hunyuan3d)
make gen-smoke ARGS="--front front.png --left left.png --back back.png"   # your own cut-out PNGs
```

`gen-smoke` writes `.data/gen/A.glb` and `.data/gen/B.glb`, and prints the GPU time of each call. The container's GPU is `PROOF3D_GEN_GPU` (default `L4`). Set `PROOF3D_GEN_GPU_USD_PER_HOUR` to its list price, so the stage records a cost. `gen-smoke` also takes `--steps`, `--octree` (surface grid), `--only-b` and `--out` (output folder). Views use our labels; the container maps them to Hunyuan3D-2mv's mirrored left and right slots (ADR 0005).

**Without a payment method on the Modal account,** GPU functions are refused ("Please add a payment method to use L4 GPU functions"; T4 too). The free credit still runs CPU containers. Set `PROOF3D_GEN_GPU=cpu` for every `gen-*` command, `gen-deploy` included:

```bash
PROOF3D_GEN_GPU=cpu make gen-weights
PROOF3D_GEN_GPU=cpu make gen-deploy
PROOF3D_GEN_GPU=cpu make gen-smoke ARGS="--front front.png --left left.png --back back.png --steps 10 --octree 256"
```

The models then run in float32 on `PROOF3D_GEN_CPU_CORES` cores (default 16), with 32 GB of memory. The setting is built into the image, because the container re-imports `modal_app.py` and would otherwise load onto CUDA. Measured on 2026-11-18 with 16 cores, 10 steps and grid 256:

| | Sampling | Surface decoding | Whole call |
|---|---|---|---|
| A (1 photo) | 30 s per step | 14.7 min | 19.9 min |
| B (3 photos) | 61 s per step | 15.1 min | 25.6 min |

Decoding grows with the cube of the grid. At the defaults (50 steps, grid 380), expect about 1.2 hours for A and 1.7 hours for B, too slow for the eval. Use 10 steps and grid 256 in the job's `generate` config on CPU, and record that the meshes were made that way.

A job builds A and B when it is imported with view labels:

```json
{"photos": [".../front.jpg", ".../left.jpg", ".../back.jpg", ".../top.jpg"],
 "views": {".../front.jpg": "front", ".../left.jpg": "left", ".../back.jpg": "back"},
 "config": {"conditions": ["A", "B"]}}
```

Ingest holds the last photo by name out (here `top.jpg`), and it is never sent to the generator. Leave C and D out of `conditions` until texture projection is built, so the job goes on to verify.

## 4. Kaggle: development runs on a T4

1. **Notebook settings:**
   - **Accelerator:** GPU T4. Not the P100: the locked torch is a CUDA 13 build, and CUDA 13 dropped Pascal cards.
   - **Internet:** on. This needs a phone-verified account.
   - **Visibility:** private.
2. **Secrets:** under Add-ons, then Secrets, add the keys from step 1. Also add `PROOF3D_GPU_USD_PER_HOUR=0`, since Kaggle time is free.
3. **Run these cells:**

```python
import os
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()
for key in [
    "PROOF3D_DATABASE_URL",
    "PROOF3D_STORAGE_BACKEND",
    "PROOF3D_S3_BUCKET",
    "PROOF3D_S3_ENDPOINT",
    "PROOF3D_S3_REGION",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "PROOF3D_GPU_USD_PER_HOUR",
]:
    os.environ[key] = secrets.get_secret(key)
```

```bash
!pip install -q uv==0.11.26
!git clone --depth 1 https://github.com/PavanSiddapuram/proof3D.git
%cd proof3D
!uv sync --frozen --no-dev --package proof3d-worker-gpu
!uv run --no-sync python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
!uv run --no-sync python -m proof3d_worker_gpu --drain
```

- **If the torch check prints `False`,** the notebook's driver is older than CUDA 13 needs. Stop there and record it in ADR 0006; don't work around it silently.
- **If the repository is private,** clone it with a fine-grained GitHub token that has read-only access to this one repository, stored as a Kaggle secret.

## What each worker reads

| Setting | Laptop (API, CPU worker) | Modal | Kaggle |
|---|---|---|---|
| Database, bucket, AWS keys | `.env` | `proof3d` secret | Kaggle secrets |
| `PROOF3D_GPU_USD_PER_HOUR` | not used | Modal's list price for the GPU | `0` |
| GPU type | n/a | `PROOF3D_MODAL_GPU`, default `L40S` | T4 |
