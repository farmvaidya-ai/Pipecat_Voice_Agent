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
from datetime import date, timedelta

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

from bot_processors.core.db_pool import close_pool, get_pool, init_pool
from bot_processors.core.voice_agent_db import init_schema
from bot_processors.pricing.agmarknet_scraper import save_scraped, scrape_all
from bot_processors.pricing.enrich_prices_with_geo import enrich as enrich_prices_with_geo
from bot_processors.pricing.export_prices_xlsx import export_to_xlsx
from bot_processors.pricing.nationwide_market_geo_reference import build_reference as build_nationwide_geo_reference
from bot_processors.pricing.price_shared import _LAST_KNOWN_PATH

_INTERVAL_SECONDS = 8 * 60 * 60  # 3x/day, matching the old Task Scheduler cadence
_NATIONWIDE_GEO_OUTPUT = str(_LAST_KNOWN_PATH.parent / "nationwide_market_geo_reference.json")

# How many past days to check for gaps each cycle, and how hard to retry a
# single date before giving up on it this cycle. Bounded lookback rather
# than scanning arbitrarily far back, so one long-standing historical gap
# (e.g. from before this backfill mechanism existed) doesn't turn into an
# ever-growing, ever-repeating backfill list every single cycle.
_BACKFILL_LOOKBACK_DAYS = 3
_MAX_ATTEMPTS_PER_DATE = 4


async def _scrape_date_with_retries(target_date: date | None) -> dict[str, dict]:
    """Runs scrape_all() for target_date (None = today), retrying with
    resume=True on failure. Confirmed live (2026-08-11): agmarknet.gov.in
    itself is flaky enough — transient form-reload failures that
    eventually cascade into scrape_all() aborting outright — that a single
    bare attempt isn't reliable for a full ~585-commodity run, for today's
    date same as any other. resume=True on the retries means each one only
    re-queries commodities the previous attempt didn't already get to
    (scrape_all() checks last_known_prices.json for an existing entry
    dated target_date), so a late retry isn't starting over from zero.
    Gives up after _MAX_ATTEMPTS_PER_DATE and returns whatever the last
    attempt managed rather than raising — one bad date shouldn't crash the
    whole daemon cycle."""
    label = target_date.isoformat() if target_date else "today"
    scraped: dict[str, dict] = {}
    for attempt in range(1, _MAX_ATTEMPTS_PER_DATE + 1):
        try:
            scraped = await asyncio.to_thread(
                scrape_all, resume=(attempt > 1), target_date=target_date
            )
            return scraped
        except Exception:
            logger.warning(
                f"⚠️ scraper_daemon: {label} attempt {attempt}/{_MAX_ATTEMPTS_PER_DATE} "
                f"failed:\n{traceback.format_exc()}"
            )
    logger.error(f"❌ scraper_daemon: {label} still failing after {_MAX_ATTEMPTS_PER_DATE} attempts, giving up this cycle")
    return scraped


async def _find_missing_recent_dates() -> list[date]:
    """Which of the last _BACKFILL_LOOKBACK_DAYS calendar days (not
    counting today, which this cycle's own regular scrape already covers)
    have zero rows in fact_market_prices — e.g. a day the daemon was down,
    or every scrape that day failed outright (this is exactly how the
    2026-08-08..10 gap happened: the site restructured its nav and every
    scrape silently failed to navigate at all)."""
    today = date.today()
    candidates = [today - timedelta(days=n) for n in range(1, _BACKFILL_LOOKBACK_DAYS + 1)]
    pool = get_pool()
    present = await pool.fetch(
        "SELECT DISTINCT arrival_date FROM fact_market_prices WHERE arrival_date = ANY($1::date[])",
        candidates,
    )
    present_dates = {row["arrival_date"] for row in present}
    return sorted(d for d in candidates if d not in present_dates)


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
                # save_scraped()'s own DB writes need. Retries internally
                # (see _scrape_date_with_retries) since the site itself is
                # flaky enough that a bare single attempt isn't reliable.
                scraped = await _scrape_date_with_retries(None)
                await save_scraped(scraped)

                # Safety net for exactly what happened 2026-08-08..10: the
                # site restructured its nav, every scrape that window
                # silently failed, and nobody noticed until asked for a
                # specific day's data days later. Cheap in the common case
                # (one DB query, no gaps) — only pays for an actual backfill
                # when there's a real hole in the last few days' data.
                missing = await _find_missing_recent_dates()
                if missing:
                    logger.info(
                        f"🕳️ scraper_daemon: no data for {len(missing)} recent day(s) "
                        f"({', '.join(d.isoformat() for d in missing)}), backfilling"
                    )
                    for d in missing:
                        backfilled = await _scrape_date_with_retries(d)
                        await save_scraped(backfilled)
                        logger.info(f"✅ scraper_daemon: backfilled {d.isoformat()} ({len(backfilled)} entries)")

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
