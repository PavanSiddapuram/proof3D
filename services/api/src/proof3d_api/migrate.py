"""Database migrations (Alembic, ADR 0007 consequences).

    python -m proof3d_api.migrate                 # upgrade to the latest schema
    python -m proof3d_api.migrate current         # show the database's revision
    python -m proof3d_api.migrate revision -m "add x"   # autogenerate from the models

The URL comes from ``PROOF3D_DATABASE_URL``. Tables are never created any other way outside
tests: ``create_tables`` is for throwaway SQLite databases only.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.migration import MigrationContext

from proof3d_api.db import make_engine
from proof3d_api.settings import get_settings

MIGRATIONS = Path(__file__).parent / "migrations"


def alembic_config(url: str) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS))
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))  # escape config interpolation
    return cfg


def upgrade(url: str, revision: str = "head") -> None:
    command.upgrade(alembic_config(url), revision)


def main(argv: Sequence[str] | None = None, url: str | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m proof3d_api.migrate")
    sub = parser.add_subparsers(dest="cmd")
    up = sub.add_parser("upgrade", help="upgrade the schema (default: to head)")
    up.add_argument("revision", nargs="?", default="head")
    sub.add_parser("current", help="show the database's current revision")
    rev = sub.add_parser("revision", help="autogenerate a migration from the models")
    rev.add_argument("-m", "--message", required=True)
    rev.add_argument("--rev-id", default=None)
    args = parser.parse_args(argv)

    url = url or get_settings().database_url
    cfg = alembic_config(url)
    if args.cmd == "current":
        with make_engine(url).connect() as conn:
            print(MigrationContext.configure(conn).get_current_revision() or "no schema yet")
    elif args.cmd == "revision":
        command.revision(cfg, message=args.message, autogenerate=True, rev_id=args.rev_id)
    else:
        command.upgrade(cfg, getattr(args, "revision", "head"))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
