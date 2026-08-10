"""Schema setup for the shared voice_agent PostgreSQL database (see
db/schema.sql). Uses the shared pool from db_pool.py — call init_pool() in
db_pool.py first (done once at Bot.py startup), then call init_schema()
below to ensure all tables/indexes exist.
"""

import os

from bot_processors.core.db_pool import get_pool

# Two levels up now (bot_processors/core/ -> project root), not one — this
# file moved a directory deeper than db_pool.py when bot_processors/ was
# split into subpackages; confirmed live this needs updating every time
# this file's own depth changes, same class of bug as the Path(__file__)
# data-file references paths.py's DATA_DIR now centralizes.
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "db", "schema.sql")


async def init_schema() -> None:
    """Creates all tables/indexes if they don't already exist. Safe to call
    every startup — every statement in schema.sql is IF NOT EXISTS."""
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        schema = f.read()
    pool = get_pool()
    async with pool.acquire() as conn:
        await conn.execute(schema)
