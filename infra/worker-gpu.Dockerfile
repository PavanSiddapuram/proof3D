# GPU worker image for the rented 24 GB+ GPU. Model packages are pinned here, by exact version
# and weight hash, once their licences are recorded in docs/licence-register.md.
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
COPY --from=ghcr.io/astral-sh/uv:0.11.26 /uv /bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON=3.12
COPY pyproject.toml uv.lock .python-version ./
COPY packages packages
COPY services services
RUN uv sync --frozen --no-dev --package proof3d-worker-gpu
ENV PATH="/app/.venv/bin:$PATH"
CMD ["python", "-m", "proof3d_worker_gpu"]
