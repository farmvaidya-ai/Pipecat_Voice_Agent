"""One-off: migrates every entry already in bot_processors/last_known_prices.json
into the new fact_market_prices table (db/schema.sql), so real scraped
history collected before the DB migration isn't lost. Safe to re-run — it's
the same upsert path agmarknet_scraper.save_scraped() uses for every live
scrape, keyed on (state, district, crop_keyword, arrival_date).

    python backfill_market_prices.py
"""

import asyncio
import json

from dotenv import load_dotenv

load_dotenv(override=True)

from bot_processors.agmarknet_scraper import save_scraped
from bot_processors.db_pool import close_pool, init_pool
from bot_processors.price_shared import _LAST_KNOWN_PATH
from bot_processors.voice_agent_db import init_schema


async def main() -> None:
    if not _LAST_KNOWN_PATH.exists():
        print(f"No file at {_LAST_KNOWN_PATH} — nothing to backfill.")
        return

    scraped = json.loads(_LAST_KNOWN_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(scraped)} entries from {_LAST_KNOWN_PATH}")

    await init_pool()
    await init_schema()  # ensures fact_market_prices exists even on a fresh DB
    await save_scraped(scraped)
    await close_pool()
    print(f"Backfilled {len(scraped)} entries into fact_market_prices.")


if __name__ == "__main__":
    asyncio.run(main())
