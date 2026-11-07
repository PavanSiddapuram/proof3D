# Engineering standards

How the repo, code, stack and product are built so a senior graphics engineer can clone it, run it and trust the results. **The bar: anyone can reproduce every number in the report from a clean checkout.**

## 1. Repository layout

One monorepo, so the pipeline, viewer, harness and evaluation always change together.

```
proof3d/
  apps/web/              React + TypeScript dashboard: import, live stages, scorecard, compare
  services/api/          FastAPI orchestrator: jobs, stages, storage, publishing
  services/worker-gpu/   GPU stages: masks, generation, projection, camera fit, scoring
  services/worker-cpu/   CPU stages: mesh checks, repair rungs 1-2, optimization, packaging
  packages/pipeline/     Stage interfaces, configs, typed inputs and outputs
  packages/geometry/     Camera fitting, texture projection, face-ID mapping, mesh checks
  packages/viewer/       Shared Three.js viewer used by the web app and phone harness
  harness/phone/         Playwright and Chrome remote-debugging phone runs
  eval/                  Dataset manifest, experiment configs, results, report scripts
  infra/                 Dockerfiles, compose file, deployment scripts
  docs/                  Architecture, ADRs, results write-up, licence register
```

Photos, meshes and renders never go in git. They live in object storage, listed in a manifest with content hashes so every run points at exact inputs.

## 2. Repository management

| Practice | Standard |
|---|---|
| Branching | Trunk-based: short branches off `main`, merged within a day or two |
| Commits | Conventional commits (`feat`, `fix`, `perf`, `eval`, `docs`) so the changelog writes itself |
| Pull requests | Template with what changed, how it was tested, and eval impact; squash merge |
| Milestones | One GitHub milestone per PRD week; every issue linked to a PRD section |
| Releases | Tag v0.1 to v0.4 at each milestone, with a short changelog and results snapshot |
| Decisions | An ADR for each big choice: multi-view model, metrics, queue, texture method |
| Licences | Code under a clear open licence; a register of every model, weight and dataset with its licence and terms |
| Secrets | Environment variables only; a committed `.env.example`; secret scanning in CI |

## 3. Code quality

**Python:** uv for environments and lockfiles; ruff for linting and formatting; mypy strict on `packages/`; Pydantic models for every config, stage input and output, and API schema; pytest, with hypothesis property tests for geometry code.

**TypeScript:** strict mode, ESLint and Prettier; Vitest for units; Playwright for end-to-end and phone runs.

**Rules that matter most**

1. Stages are pure and typed. Each takes typed inputs, writes outputs to storage and returns a typed result. No hidden state.
2. Config, not code edits. Experiments change YAML configs validated by Pydantic, never constants in code.
3. Determinism. Fixed seeds; model versions and weight hashes pinned; GPU nondeterminism documented where it remains.
4. No notebooks in the main path. Anything that produces a reported number is a script in `eval/`.
5. Every stage measures itself: wall time, GPU-seconds and peak memory, stored with the job.
6. Pre-commit hooks run formatting, linting and type checks before every commit.

## 4. Testing strategy

| Level | What it proves | Example |
|---|---|---|
| Unit and property | Geometry code is correct | Render a known mesh from a known camera; camera fitting recovers it within tolerance |
| Golden | Results don't drift silently | A fixed product's scores stay within tolerance of stored values |
| Integration | Stages connect | The full chain runs on a tiny mesh on CPU in CI |
| End to end | The product works | URL to published Shopify product on the demo store, run nightly |
| Eval regression | Changes don't make accuracy worse | A 5-product subset rescored on pipeline changes; fail if any metric drops beyond tolerance |
| Performance regression | Models stay phone-ready | Phone harness thresholds checked on each release |

Coverage target: 80% or more for `packages/geometry` and `packages/pipeline`. Thin wrappers around third-party models need integration tests, not coverage numbers.

## 5. Tech stack

| Layer | Choice | Why |
|---|---|---|
| Web app | React, TypeScript, Vite, Three.js | Matches the target stack |
| API | Python, FastAPI, Pydantic | The ML stack is Python; typed schemas end to end |
| Database | Postgres | Jobs, scores, versions and device runs are relational |
| Queue | Postgres-backed job queue | One less service to run; enough for one GPU |
| GPU stages | PyTorch; nvdiffrast for rendering and camera fitting | Differentiable rendering is needed for fitting and projection |
| Models | BiRefNet (masks), Hunyuan3D-2mv (multi-view), TRELLIS.2 (single-photo baseline) | Strong open candidates; licences checked first |
| Metrics | LPIPS, DINOv2, FLIP, silhouette IoU | Perceptual, semantic, pixel and shape coverage |
| Geometry | trimesh, xatlas, meshoptimizer | Mesh handling, UV unwrapping, simplification and LODs |
| Packaging | glTF Transform with KTX2 and meshopt compression | Standard web delivery formats |
| Storage | S3-compatible object storage | Cheap, versioned asset storage |
| Compute | One rented 24 GB+ GPU, by the hour | Enough for the POC; no idle cost |
| CI | GitHub Actions | Lint, types, tests and eval regression on every PR |
| Errors and logs | Structured JSON logs, a per-job trace in the database, Sentry | Debug any failed job from its record |
| Local setup | Docker Compose | One command to run everything except the GPU |

## 6. Product quality

- **Live progress:** every stage shows status, time taken and cost as it runs.
- **Readable scorecard:** one headline accuracy score, then per-metric detail, error heatmaps and unverified surfaces highlighted on the model.
- **Honest comparison:** single-photo and Proof3D side by side with the held-out real photo, from the same camera.
- **Useful errors:** "No usable back view found; 31% of the surface is unverified" instead of a stack trace.
- **Recoverable:** rerun from any stage with one click.
- **Fast viewer:** the published model meets the phone targets in the PRD.
- **No invented numbers:** every number on screen traces to a stored measurement.

## 7. Reproducibility and observability

Every result is tagged with: git commit and pipeline version; model names, versions and weight hashes; config hash and random seed; dataset manifest version; GPU type and driver.

One command, `make report`, regenerates every table and chart in the write-up from stored results. Any job can be inspected stage by stage from the web app.

## 8. Documentation

| Document | Contents |
|---|---|
| README | What it is, a demo GIF, and a quickstart that runs in under 10 minutes |
| Architecture | The pipeline diagram, components, data model and API |
| ADRs | Each major decision, the options considered and why |
| Results | Method, dataset, all four conditions, failures included |
| Licence register | Every model, weight and dataset with its terms |
| Demo video | Two minutes, URL to published product on a budget phone |

## 9. Security and data ethics

- Use only products photographed yourself, in your own demo store.
- Shopify access token with the smallest scopes needed; stored as a secret.
- No secrets in the repo; CI fails on detected secrets.
- Respect every model's licence and territory terms before any public demo.

## 10. Achievement levels

| Level | What is true |
|---|---|
| 1. Working | URL to published 3D runs end to end on 20 products; conditions A and C scored; results table published |
| 2. Strong | All four conditions; repair rungs 1 and 2; phone targets met; CI with eval regression; every number reproducible with one command |
| 3. Exceptional | Fewer-photo study; human agreement study; under 10 minutes and a known cost per product; public write-up and live demo link; a stranger can run the repo from the README |

**Definition of done for any stage:** typed interface, tests passing, timings and costs recorded, documented in the architecture doc, and its effect visible in the scorecard.

What a reviewer will likely check first: the README quickstart, the results table including failures, the camera-fitting and projection code, and whether the phone demo matches the numbers. Make those four flawless.
