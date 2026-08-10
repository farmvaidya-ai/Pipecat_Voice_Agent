"""Always-on background loop that periodically runs bot_processors.pricing.agmarknet_scraper.

Replaces a Windows Task Scheduler entry, which on this machine kept getting
auto-disabled after each run (most likely by one of the third-party AV
products installed alongside Windows Defender — scheduled tasks that launch a
headless browser look like malware persistence to that kind of heuristic).
A plain long-running process sidesteps that entirely.

Run with:  python -m bot_processors.pricing.scraper_daemon
Stop with: close its console window, or `taskkill /PID <pid> /F`.
"""

import asyncio
import traceback

from dotenv import load_dotenv
from loguru import logger

# Must run before any bot_processors import below — db_pool.py reads
# DATABASE_URL from the environment at MODULE IMPORT time (a top-level
# statement, not inside a function), so .env has to already be loaded by
# the time that import executes. Bot.py does this same load_dotenv() call
# itself before its own bot_processors imports for the identical reason;
# this daemon is a separate entry point (own OS process, own `python -m`
# invocation) so it needs its own copy rather than relying on Bot.py's.
# Confirmed live (2026-08-10): without this, DATABASE_URL comes back empty
# and init_pool() raises immediately on startup.
load_dotenv(override=True)

from bot_processors.core.db_pool import close_pool, init_pool
from bot_processors.core.voice_agent_db import init_schema
from bot_processors.pricing.agmarknet_scraper import save_scraped, scrape_all
from bot_processors.pricing.enrich_prices_with_geo import enrich as enrich_prices_with_geo
from bot_processors.pricing.export_prices_xlsx import export_to_xlsx
from bot_processors.pricing.nationwide_market_geo_reference import build_reference as build_nationwide_geo_reference
from bot_processors.pricing.price_shared import _LAST_KNOWN_PATH

_INTERVAL_SECONDS = 8 * 60 * 60  # 3x/day, matching the old Task Scheduler cadence
_NATIONWIDE_GEO_OUTPUT = str(_LAST_KNOWN_PATH.parent / "nationwide_market_geo_reference.json")


async def run_forever() -> None:
    # This daemon is its own OS process, separate from Bot.py — it needs its
    # own pool (save_scraped writes to fact_market_prices via asyncpg) and
    # its own schema check (harmless/no-op once the bot's own startup has
    # already run it, since every statement in schema.sql is IF NOT EXISTS).
    await init_pool()
    await init_schema()
    try:
        while True:
            try:
                logger.info("⏰ scraper_daemon: starting scrape run")
                # scrape_all() drives a real (sync Playwright) browser — runs
                # in a thread so this coroutine doesn't block the event loop
                # save_scraped()'s own DB writes need.
                scraped = await asyncio.to_thread(scrape_all)
                await save_scraped(scraped)
                # export_to_xlsx still reads/writes the JSON export path for
                # now (a human-readable spreadsheet snapshot, not something
                # the bot's own lookups depend on) — unaffected by
                # last_known_prices.json's retirement as the bot's data
                # source.
                count = export_to_xlsx(_LAST_KNOWN_PATH.with_suffix(".xlsx"))
                logger.info(f"📊 scraper_daemon: refreshed xlsx export ({count} rows)")

                # A scrape can surface market yards last_known_prices.json has
                # never seen before, leaving them with no coordinates until
                # someone geocodes them — this used to be a manual, easy-to-forget
                # step (rerun nationwide_market_geo_reference.py then
                # enrich_prices_with_geo.py by hand). Doing it here instead keeps
                # location_lookup.py's nearby-market search current automatically:
                # build_reference() only geocodes triples its cache hasn't seen
                # before, so this is a no-op fast path on every run with no new
                # markets, and location_lookup.py already reloads its in-memory
                # market table whenever the reference file's mtime changes — no
                # live geocoding on the call path, ever.
                logger.info("🌍 scraper_daemon: checking for ungeocoded market yards")
                await asyncio.to_thread(build_nationwide_geo_reference, _NATIONWIDE_GEO_OUTPUT)
                await asyncio.to_thread(enrich_prices_with_geo)
            except Exception:
                logger.error(f"❌ scraper_daemon: scrape run failed:\n{traceback.format_exc()}")
            logger.info(f"💤 scraper_daemon: sleeping {_INTERVAL_SECONDS}s until next run")
            await asyncio.sleep(_INTERVAL_SECONDS)
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(run_forever())
