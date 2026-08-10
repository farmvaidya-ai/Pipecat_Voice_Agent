"""TRUNCATEs every caller/call table in the voice_agent database (dim_contacts
+ all fact_* tables except fact_market_prices, which isn't caller data) —
permanently deletes all caller/call history. Reused often during testing to
reset to a clean slate; not a one-off despite the name.

    python scripts/wipe_db.py
"""

import asyncio
import sys
from pathlib import Path

# Run as a plain script (not -m), so the project root — where the
# bot_processors package lives — isn't on sys.path by default the way it
# is for python -m scripts.wipe_db; add it explicitly.
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv(override=True)

from bot_processors.core.db_pool import close_pool, init_pool, get_pool

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
