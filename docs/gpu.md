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

Ingest and verify already work in their CPU form: masks come from photo transparency, and only silhouettes are scored. They need no GPU, so `make worker-gpu-local` runs them on your laptop. Modal and Kaggle are needed once generation, rendering and appearance scoring are built.

## 2. Modal: TRELLIS.2 and scored eval runs

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

## 3. Kaggle: development runs on a T4

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

