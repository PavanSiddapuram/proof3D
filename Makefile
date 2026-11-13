.PHONY: setup lint typecheck test cov up down api gpu-drain report

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

api:
	uv run uvicorn proof3d_api.main:create_app --factory --reload

gpu-drain:        ## Process queued GPU stages on Modal, then exit (docs/gpu.md). ARGS="--max-stages 1"
	uv run --with modal==1.6.3 modal run services/worker-gpu/modal_app.py $(ARGS)

report:           ## Regenerate every table and chart in the write-up from stored results
	uv run python eval/scripts/make_report.py
