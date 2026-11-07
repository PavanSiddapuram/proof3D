FROM python:3.12.13-slim
COPY --from=ghcr.io/astral-sh/uv:0.11.26 /uv /bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock .python-version ./
COPY packages packages
COPY services services
RUN uv sync --frozen --no-dev --package proof3d-worker-cpu --package proof3d-api
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
CMD ["uvicorn", "proof3d_api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
