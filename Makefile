.PHONY: setup lint typecheck test cov up down migrate api worker-gpu-local demo gpu-drain gen-weights gen-deploy gen-smoke report

setup:            ## Install the Python workspace and git hooks
	uv sync
	uv run pre-commit install

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy

test:
	uv run pytest

cov:              ## Coverage gate: 80% on packages/geometry and packages/pipeline (standards 4)
	uv run pytest packages --cov=proof3d_geometry --cov=proof3d_pipeline --cov-report=term-missing --cov-fail-under=80

up:               ## Postgres + API locally (everything except the GPU)
	docker compose -f infra/docker-compose.yml up -d --build

down:
	docker compose -f infra/docker-compose.yml down

migrate:          ## Bring the database schema up to date (PROOF3D_DATABASE_URL)
	uv run python -m proof3d_api.migrate

api:
	uv run uvicorn proof3d_api.main:create_app --factory --reload

worker-gpu-local: ## Run queued GPU-worker stages on this machine (ingest, verify run on CPU), then exit
	PYTHONPATH=services/worker-gpu/src uv run python -m proof3d_worker_gpu --drain

demo:             ## Upload, import and verify a synthetic model through the running API. ARGS="--wrong"
	uv run python eval/scripts/demo_verify.py $(ARGS)

gpu-drain:        ## Process queued GPU stages on Modal, then exit (docs/gpu.md). ARGS="--max-stages 1"
	uv run --with modal==1.6.3 modal run services/worker-gpu/modal_app.py $(ARGS)

GEN_APP := services/gen-hunyuan3d/modal_app.py

gen-weights:      ## Download the pinned Hunyuan3D weights into the Modal volume, once
	uv run --with modal==1.6.3 modal run $(GEN_APP)::download_weights

gen-deploy:       ## Deploy the shape-generation container that generate_shape calls (ADR 0009)
	uv run --with modal==1.6.3 modal deploy $(GEN_APP)

gen-smoke:        ## One A and one B call on your own cut-out PNGs. ARGS="--front f.png --left l.png"
	uv run --with modal==1.6.3 modal run $(GEN_APP)::smoke $(ARGS)

report:           ## Regenerate every table and chart in the write-up from stored results
	uv run python eval/scripts/make_report.py
