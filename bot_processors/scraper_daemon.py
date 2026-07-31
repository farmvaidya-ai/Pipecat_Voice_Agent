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
from bot_processors.export_prices_xlsx import export_to_xlsx
from bot_processors.price_shared import _LAST_KNOWN_PATH

_INTERVAL_SECONDS = 8 * 60 * 60  # 3x/day, matching the old Task Scheduler cadence


def run_forever() -> None:
    while True:
        try:
            logger.info("⏰ scraper_daemon: starting scrape run")
            save_scraped(scrape_all())
            count = export_to_xlsx(_LAST_KNOWN_PATH.with_suffix(".xlsx"))
            logger.info(f"📊 scraper_daemon: refreshed xlsx export ({count} rows)")
        except Exception:
            logger.error(f"❌ scraper_daemon: scrape run failed:\n{traceback.format_exc()}")
        logger.info(f"💤 scraper_daemon: sleeping {_INTERVAL_SECONDS}s until next run")
        time.sleep(_INTERVAL_SECONDS)


if __name__ == "__main__":
    run_forever()
