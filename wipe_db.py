"""One-off: TRUNCATEs every table in the voice_agent database (dim_contacts +
all fact_* tables), permanently deleting all caller/call data. Run once,
then delete this file — it is not meant to be a permanent part of the repo.

    python wipe_db.py
"""

import asyncio

from dotenv import load_dotenv

load_dotenv(override=True)

from bot_processors.db_pool import close_pool, init_pool, get_pool

_TABLES = [
    "fact_performance",
    "fact_toolcalls",
    "fact_conversations",
    "fact_sessions",
    "fact_conversation_summary",
    "dim_contacts",
]


async def main() -> None:
    await init_pool()
    pool = get_pool()
    async with pool.acquire() as conn:
        await conn.execute(f"TRUNCATE TABLE {', '.join(_TABLES)} RESTART IDENTITY CASCADE;")
    await close_pool()
    print(f"Truncated: {', '.join(_TABLES)}")


if __name__ == "__main__":
    asyncio.run(main())
