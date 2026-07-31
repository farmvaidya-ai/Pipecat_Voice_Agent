"""Shared PostgreSQL connection pool, used by both caller_db.py and
voice_agent_db.py. Unlike SQLite (a cheap per-call file connection), Postgres
is a real server — opening a fresh connection on every call is wasteful and
slow, so the pool is created once at process startup (see Bot.py's main())
and reused for the life of the process.
"""

import os

import asyncpg
from loguru import logger

DATABASE_URL = os.getenv("DATABASE_URL", "")

_pool: asyncpg.Pool | None = None


async def init_pool() -> None:
    """Creates the shared pool. Call once at startup before any call comes
    in (see Bot.py's main()). Safe to call more than once — a no-op if the
    pool already exists."""
    global _pool
    if _pool is not None:
        return
    if not DATABASE_URL:
        raise ValueError("[db_pool] DATABASE_URL is not set in .env")
    _pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=10)
    logger.info("[db_pool] PostgreSQL connection pool created")


def get_pool() -> asyncpg.Pool:
    """Returns the shared pool. Raises if init_pool() hasn't run yet —
    every caller runs inside the pipeline, which only starts after main()
    has already awaited init_pool()."""
    if _pool is None:
        raise RuntimeError("[db_pool] Pool not initialized — call init_pool() first")
    return _pool


async def close_pool() -> None:
    """Closes the shared pool on shutdown. Safe to call even if the pool
    was never created."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("[db_pool] PostgreSQL connection pool closed")
