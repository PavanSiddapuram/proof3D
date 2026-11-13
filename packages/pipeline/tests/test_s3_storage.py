from __future__ import annotations

import io
from typing import Any

import pytest

from proof3d_pipeline.storage import S3Storage


class _ClientError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class FakeS3:
    """In-memory stand-in for boto3's client: same keyword calls, same error shape."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.head_error: str | None = None

    def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> dict[str, Any]:
        self.objects[(Bucket, Key)] = Body
        return {}

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        if (Bucket, Key) not in self.objects:
            raise _ClientError("NoSuchKey")
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        if self.head_error:
            raise _ClientError(self.head_error)
        if (Bucket, Key) not in self.objects:
            raise _ClientError("404")
        return {}


def test_roundtrip_with_prefix() -> None:
    client = FakeS3()
    store = S3Storage("bucket", client, prefix="/runs/")
    assert store.put("jobs/1/a.json", b"{}") == "jobs/1/a.json"
    assert ("bucket", "runs/jobs/1/a.json") in client.objects
    assert store.get("jobs/1/a.json") == b"{}"
    assert store.exists("jobs/1/a.json")
    assert not store.exists("jobs/1/missing.json")


def test_no_prefix_uses_key_as_is() -> None:
    client = FakeS3()
    S3Storage("bucket", client).put("a/b", b"x")
    assert ("bucket", "a/b") in client.objects


@pytest.mark.parametrize("key", ["", "/abs", "a/../b", "..", "a" + chr(92) + "b"])
def test_rejects_unsafe_keys(key: str) -> None:
    with pytest.raises(ValueError, match="invalid storage key"):
        S3Storage("bucket", FakeS3()).put(key, b"x")


def test_exists_reraises_errors_other_than_missing() -> None:
    client = FakeS3()
    client.head_error = "AccessDenied"
    with pytest.raises(_ClientError):
        S3Storage("bucket", client).exists("a")


def test_requires_bucket() -> None:
    with pytest.raises(ValueError, match="bucket"):
        S3Storage("", FakeS3())


def test_from_env_builds_a_boto3_client(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("boto3")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    store = S3Storage.from_env("bucket", endpoint_url="https://s3.example.com", region="auto")
    assert store.client.meta.endpoint_url == "https://s3.example.com"  # type: ignore[attr-defined]
