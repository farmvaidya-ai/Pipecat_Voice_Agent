"""Constants shared between bot_processors.pricing.price_lookup (reads the cache,
drives the live conversation) and bot_processors.pricing.agmarknet_scraper (writes
the cache, on a schedule and on demand).

Split out purely to avoid a circular import: price_lookup's live-fallback
path needs to call agmarknet_scraper.scrape_single/etc, and agmarknet_scraper
needs the cache file's path/key format — if either module imported the other
directly for these, the two imports would form a cycle.
"""

from bot_processors.paths import DATA_DIR

_LAST_KNOWN_PATH = DATA_DIR / "last_known_prices.json"
_LAST_KNOWN_KEY_SEP = "||"


# Shared commodity-name -> cache-key normalization. Both
# bot_processors.pricing.price_lookup (resolving a caller's spoken commodity to a
# lookup key) and bot_processors.pricing.agmarknet_scraper (writing that same key
# when it scrapes a commodity) must derive identical keys from identical
# commodity names, or the daemon's freshly scraped rows would silently
# never be found by a live call's lookup — kept here as the one definition
# both sides import instead of risking two copies drifting apart.
def normalize_commodity_name(name: str) -> str:
    return name.split("(")[0].strip().lower()


def crop_keyword_for(name: str) -> str:
    return normalize_commodity_name(name).replace(" ", "_").replace("/", "_")
