"""One-off: migrates every entry already in bot_processors/data/last_known_prices.json
into the fact_market_prices table (db/schema.sql), so real scraped
history collected before the DB migration isn't lost. Safe to re-run — it's
the same upsert path agmarknet_scraper.save_scraped() uses for every live
scrape, keyed on (state, district, crop_keyword, arrival_date). Already run
once (2026-08-10, all 6579 entries) — kept for reference/disaster recovery,
not something that needs running again in the normal course of things.

    python scripts/backfill_market_prices.py
"""

import asyncio
import json
import sys
from pathlib import Path

# Run as a plain script (not -m), so the project root — where the
# bot_processors package lives — isn't on sys.path by default the way it
# is for python -m scripts.backfill_market_prices; add it explicitly.
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv(override=True)

from bot_processors.core.db_pool import close_pool, init_pool
from bot_processors.core.voice_agent_db import init_schema
from bot_processors.pricing.agmarknet_scraper import save_scraped
from bot_processors.pricing.price_shared import _LAST_KNOWN_PATH


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
