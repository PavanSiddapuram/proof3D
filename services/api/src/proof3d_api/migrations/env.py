"""Alembic environment. Run migrations with ``python -m proof3d_api.migrate``, not ``alembic``,
so the database URL comes from settings (environment variables) like everywhere else."""

from __future__ import annotations

from typing import Any

from alembic import context

from proof3d_api.db import Base, make_engine

config = context.config
url = config.get_main_option("sqlalchemy.url")
assert url, "proof3d_api.migrate sets sqlalchemy.url"

# render_as_batch: SQLite cannot ALTER most things, so Alembic rebuilds the table instead.
options: dict[str, Any] = {
    "target_metadata": Base.metadata,
    "render_as_batch": True,
    "compare_type": True,
}

if context.is_offline_mode():
    context.configure(url=url, literal_binds=True, **options)
    with context.begin_transaction():
        context.run_migrations()
else:
    with make_engine(url).connect() as connection:
        context.configure(connection=connection, **options)
        with context.begin_transaction():
            context.run_migrations()
