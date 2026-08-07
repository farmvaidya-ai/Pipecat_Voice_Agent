"""Always-on background loop that periodically runs bot_processors.agmarknet_scraper.

Replaces a Windows Task Scheduler entry, which on this machine kept getting
auto-disabled after each run (most likely by one of the third-party AV
products installed alongside Windows Defender — scheduled tasks that launch a
headless browser look like malware persistence to that kind of heuristic).
A plain long-running process sidesteps that entirely.

Run with:  python -m bot_processors.scraper_daemon
Stop with: close its console window, or `taskkill /PID <pid> /F`.
"""

import time
import traceback

from loguru import logger

from bot_processors.agmarknet_scraper import save_scraped, scrape_all
from bot_processors.enrich_prices_with_geo import enrich as enrich_prices_with_geo
from bot_processors.export_prices_xlsx import export_to_xlsx
from bot_processors.nationwide_market_geo_reference import build_reference as build_nationwide_geo_reference
from bot_processors.price_shared import _LAST_KNOWN_PATH

_INTERVAL_SECONDS = 8 * 60 * 60  # 3x/day, matching the old Task Scheduler cadence
_NATIONWIDE_GEO_OUTPUT = str(_LAST_KNOWN_PATH.parent / "nationwide_market_geo_reference.json")


def run_forever() -> None:
    while True:
        try:
            logger.info("⏰ scraper_daemon: starting scrape run")
            save_scraped(scrape_all())
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
            build_nationwide_geo_reference(_NATIONWIDE_GEO_OUTPUT)
            enrich_prices_with_geo()
        except Exception:
            logger.error(f"❌ scraper_daemon: scrape run failed:\n{traceback.format_exc()}")
        logger.info(f"💤 scraper_daemon: sleeping {_INTERVAL_SECONDS}s until next run")
        time.sleep(_INTERVAL_SECONDS)


if __name__ == "__main__":
    run_forever()
