"""POST /uploads, and import refusing files that were never uploaded."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from proof3d_api.main import create_app
from proof3d_api.settings import Settings
from proof3d_pipeline.storage import LocalStorage

PNG = b"\x89PNG\r\n\x1a\n" + b"\0" * 32
JPEG = b"\xff\xd8\xff\xe0" + b"\0" * 32
GLB = b"glTF" + b"\0" * 32


@pytest.fixture
def storage(tmp_path: Path) -> LocalStorage:
    return LocalStorage(tmp_path / "store")


@pytest.fixture
def client(storage: LocalStorage) -> Iterator[TestClient]:
    settings = Settings(database_url="sqlite+pysqlite:///:memory:", max_upload_mb=1, _env_file=None)  # type: ignore[call-arg]
    with TestClient(create_app(settings, create_tables=True, storage=storage)) as c:
        yield c


def _upload(client: TestClient, name: str, data: bytes) -> tuple[int, dict[str, object]]:
    r = client.post("/uploads", params={"filename": name}, content=data)
    return r.status_code, r.json()


@pytest.mark.parametrize(
    ("name", "data", "kind", "stored_as"),
    [
        ("front.png", PNG, "image", "front.png"),
        ("Front View (1).JPEG", JPEG, "image", "Front_View_1.jpg"),
        ("model.glb", GLB, "model", "model.glb"),
        ("photo.glb", PNG, "image", "photo.png"),  # the bytes decide the type, not the name
        ("C:\\Users\\me\\..\\x.png", PNG, "image", "x.png"),
        ("...", PNG, "image", "upload.png"),
    ],
)
def test_upload_stores_content_addressed_files(
    client: TestClient, storage: LocalStorage, name: str, data: bytes, kind: str, stored_as: str
) -> None:
    status, body = _upload(client, name, data)
    assert status == 201
    assert body["kind"] == kind
    assert body["bytes"] == len(data)
    key = str(body["key"])
    assert key.startswith("uploads/")
    assert key.endswith("/" + stored_as)
    assert storage.get(key) == data


def test_same_file_gives_same_key(client: TestClient) -> None:
    assert _upload(client, "a.png", PNG)[1]["key"] == _upload(client, "a.png", PNG)[1]["key"]


def test_upload_rejects_unknown_types_and_oversize(client: TestClient) -> None:
    assert _upload(client, "notes.txt", b"hello")[0] == 415
    status, body = _upload(client, "big.png", PNG + b"\0" * 2**20)
    assert status == 413
    assert body["detail"] == "uploads are limited to 1 MB"


def test_import_refuses_files_not_in_storage(client: TestClient) -> None:
    key = str(_upload(client, "front.png", PNG)[1]["key"])
    ok = client.post("/products/import", json={"photos": [key]})
    assert ok.status_code == 201
    for missing in ["uploads/nope/front.png", "../../etc/passwd"]:
        r = client.post("/products/import", json={"photos": [key, missing]})
        assert r.status_code == 422
        assert missing in r.json()["detail"]
    glb = {"glb": "uploads/nope/model.glb", "generator": "x", "held_out": [key]}
    r = client.post("/products/import", json={"photos": [key], "model": glb})
    assert r.status_code == 422
