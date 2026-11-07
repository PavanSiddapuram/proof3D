from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration from environment variables only (standards 2, Secrets). See .env.example."""

    model_config = SettingsConfigDict(env_prefix="PROOF3D_", env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://proof3d:proof3d@localhost:5432/proof3d"
    storage_root: str = "./.data/storage"
    pipeline_version: str = "0.1.0"
    max_stage_attempts: int = 3
    # Create tables on startup. Fine until the schema first changes under real data; then
    # switch to Alembic migrations and turn this off.
    create_tables: bool = True
    gpu_usd_per_hour: float = 0.0  # set to the rented GPU's price so costs are real, not invented

    shopify_store_domain: str | None = None
    shopify_admin_token: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
