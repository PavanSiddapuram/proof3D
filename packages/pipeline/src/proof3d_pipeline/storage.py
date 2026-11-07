"""Object storage behind one small interface. Local disk for dev and CI; S3 in deployment."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol


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
