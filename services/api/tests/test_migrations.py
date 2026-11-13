"""Migrations and models must describe the same schema."""

from __future__ import annotations

from pathlib import Path

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import inspect

from proof3d_api.db import Base, make_engine
from proof3d_api.migrate import main, upgrade


def test_migrations_match_the_models(tmp_path: Path) -> None:
    url = f"sqlite+pysqlite:///{tmp_path / 'm.sqlite'}"
    upgrade(url)
    engine = make_engine(url)
    with engine.connect() as conn:
        diff = compare_metadata(
            MigrationContext.configure(conn, opts={"compare_type": True}), Base.metadata
        )
    assert diff == [], f"models changed without a migration; run migrate revision: {diff}"
    assert "planned_skip" in {c["name"] for c in inspect(engine).get_columns("stage_runs")}


def test_cli_upgrades_and_reports_revision(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    url = f"sqlite+pysqlite:///{tmp_path / 'cli.sqlite'}"
    main(["current"], url=url)
    assert "no schema yet" in capsys.readouterr().out
    assert main([], url=url) == 0
    main(["current"], url=url)
    assert capsys.readouterr().out.strip() == "0001"
