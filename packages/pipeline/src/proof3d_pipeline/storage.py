"""Object storage behind one small interface.

``LocalStorage`` is for development and CI. ``S3Storage`` is for any S3-compatible bucket (AWS,
Cloudflare R2, Backblaze B2, MinIO). It is the shared store when the GPU worker runs somewhere
else, on Modal or Kaggle, and must hand files to the API and the CPU worker (ADR 0006).
"""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
from typing import Any, Protocol


class Storage(Protocol):
    def put(self, key: str, data: bytes) -> str: ...
    def get(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...


def content_key(prefix: str, data: bytes, suffix: str = "") -> str:
    """Content-addressed key so identical outputs dedupe and manifests can pin exact inputs."""
    return f"{prefix.rstrip('/')}/{hashlib.sha256(data).hexdigest()}{suffix}"


class LocalStorage:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError(f"key escapes storage root: {key!r}")
        return path

    def put(self, key: str, data: bytes) -> str:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
        return key

    def get(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).exists()


class S3Client(Protocol):
    """The three calls ``S3Storage`` makes; boto3's client satisfies it."""

    def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> Any: ...
    def get_object(self, *, Bucket: str, Key: str) -> Any: ...
    def head_object(self, *, Bucket: str, Key: str) -> Any: ...


_MISSING_CODES = frozenset({"404", "NoSuchKey", "NotFound"})


class S3Storage:
    """Keys map to ``prefix/key`` in one bucket. The client is injected so tests need no network."""

    def __init__(self, bucket: str, client: S3Client, prefix: str = "") -> None:
        if not bucket:
            raise ValueError("S3Storage needs a bucket name")
        self.bucket = bucket
        self.client = client
        self.prefix = prefix.strip("/")

    @classmethod
    def from_env(
        cls,
        bucket: str,
        endpoint_url: str | None = None,
        region: str | None = None,
        prefix: str = "",
    ) -> S3Storage:
        """Build a boto3 client. Credentials come from the standard AWS_* environment variables."""
        import boto3

        client = boto3.client("s3", endpoint_url=endpoint_url or None, region_name=region or None)
        return cls(bucket, client, prefix)

    def _key(self, key: str) -> str:
        parts = PurePosixPath(key).parts
        if not key or key.startswith("/") or ".." in parts or "\\" in key:
            raise ValueError(f"invalid storage key: {key!r}")
        return f"{self.prefix}/{key}" if self.prefix else key

    def put(self, key: str, data: bytes) -> str:
        self.client.put_object(Bucket=self.bucket, Key=self._key(key), Body=data)
        return key

    def get(self, key: str) -> bytes:
        response = self.client.get_object(Bucket=self.bucket, Key=self._key(key))
        body: bytes = response["Body"].read()
        return body

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=self._key(key))
        except Exception as exc:
            code = str(getattr(exc, "response", {}).get("Error", {}).get("Code", ""))
            if code in _MISSING_CODES:
                return False
            raise
        return True
