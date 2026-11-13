from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

from proof3d_pipeline.storage import LocalStorage, S3Storage, Storage


class Settings(BaseSettings):
    """Configuration from environment variables only (standards 2, Secrets). See .env.example."""

    model_config = SettingsConfigDict(env_prefix="PROOF3D_", env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://proof3d:proof3d@localhost:5432/proof3d"
    pipeline_version: str = "0.1.0"
    max_stage_attempts: int = 3
    # Create tables on startup, for throwaway SQLite databases in tests only. Real databases
    # get their schema from migrations: python -m proof3d_api.migrate
    create_tables: bool = False
    max_upload_mb: int = 50
    gpu_usd_per_hour: float = 0.0  # set to the rented GPU's price so costs are real, not invented

    # "local" for one machine. "s3" whenever a worker runs elsewhere (Modal, Kaggle), so every
    # process reads and writes the same files (ADR 0006). S3 credentials are the AWS_* variables.
    storage_backend: Literal["local", "s3"] = "local"
    storage_root: str = "./.data/storage"
    s3_bucket: str = ""
    s3_endpoint: str = ""
    s3_region: str = ""
    s3_prefix: str = ""

    shopify_store_domain: str | None = None
    shopify_admin_token: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()


def make_storage(settings: Settings) -> Storage:
    if settings.storage_backend == "s3":
        if not settings.s3_bucket:
            raise ValueError("PROOF3D_STORAGE_BACKEND=s3 needs PROOF3D_S3_BUCKET")
        return S3Storage.from_env(
            settings.s3_bucket, settings.s3_endpoint, settings.s3_region, settings.s3_prefix
        )
    return LocalStorage(settings.storage_root)
