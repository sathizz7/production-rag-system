from __future__ import annotations

import os

from sqlalchemy import engine_from_config, pool

from alembic import context
from rag.config import get_settings
from rag.db import metadata_obj

config = context.config

# Override C: resolve URL from env (set by run_migrations), else from Settings (.env)
url = os.environ.get("DATABASE_URL") or get_settings().database_url
config.set_main_option("sqlalchemy.url", url)

target_metadata = metadata_obj


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


run_migrations_online()
