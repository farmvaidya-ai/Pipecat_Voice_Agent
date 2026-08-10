"""Builds a (state, district, market) -> (latitude, longitude) reference
table covering every market yard scraped anywhere in India, by geocoding the
unique (state, district, market) triples found directly in
last_known_prices.json.

Unlike pincode_geo_reference.py / market_yard_geo_reference.py (built from
AP/Telangana-specific source sheets that needed district-name reconciliation),
last_known_prices.json's own district field IS Agmarknet's current district
name for every state -- there's no separate reference file or crosswalk
needed nationwide, just a direct geocode per unique market.

CLI: python -m bot_processors.pricing.nationwide_market_geo_reference
"""

import json
import math
import re
import sys
import time
from collections import defaultdict

from loguru import logger

from bot_processors.paths import DATA_DIR

_LAST_KNOWN_PATH = DATA_DIR / "last_known_prices.json"
_CACHE_PATH = DATA_DIR / "nationwide_market_geo_cache.json"
_NOMINATIM_USER_AGENT = "farm_vaidya_pincode_geocoder"
_OUTLIER_DISTANCE_KM = 150

# Same suffix-noise pattern confirmed across both AP pipelines: parenthetical
# alt-names, trailing comma-attached codes, and the generic market-type
# suffix all break Nominatim matching for otherwise-valid town names.
_PAREN_RE = re.compile(r"\([^)]*\)")
_MARKET_SUFFIX_RE = re.compile(
    r"\s*\b(apmc|market|mandi|rbz|rythu bazar)\b\.?\s*$", re.IGNORECASE
)


def _clean_market_for_query(market: str) -> str:
    cleaned = _PAREN_RE.sub("", market)
    cleaned = cleaned.split(",")[0]
    cleaned = _MARKET_SUFFIX_RE.sub("", cleaned).strip()
    return cleaned or market


def _unique_triples() -> list[tuple[str, str, str]]:
    with open(_LAST_KNOWN_PATH, encoding="utf-8") as f:
        data = json.load(f)
    triples = {(v["state"], v["district"], v["market"]) for v in data.values()}
    return sorted(triples)


def _load_cache() -> dict:
    if _CACHE_PATH.exists():
        with open(_CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict) -> None:
    with open(_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def geocode_markets(triples: list[tuple[str, str, str]], cache: dict) -> dict:
    from geopy.geocoders import Nominatim

    geolocator = Nominatim(user_agent=_NOMINATIM_USER_AGENT)
    todo = [t for t in triples if f"{t[0]}|{t[1]}|{t[2]}" not in cache or "error" in cache[f"{t[0]}|{t[1]}|{t[2]}"]]
    already_ok = len(triples) - len(todo)
    logger.info(f"Geocoding {len(todo)} markets nationwide ({already_ok} already resolved and cached)")

    for i, (state, district, market) in enumerate(todo, 1):
        key = f"{state}|{district}|{market}"
        cleaned = _clean_market_for_query(market)
        queries = [f"{cleaned}, {district}, {state}, India"] if district else []
        queries.append(f"{cleaned}, {state}, India")

        loc = None
        last_err = None
        for query in queries:
            try:
                loc = geolocator.geocode(query, exactly_one=True, timeout=15)
            except Exception as e:
                last_err = e
                loc = None
            time.sleep(1)
            if loc is not None:
                break

        if loc is not None:
            cache[key] = {"latitude": loc.latitude, "longitude": loc.longitude}
        else:
            cache[key] = {"error": str(last_err) if last_err else "no result"}

        if i % 25 == 0 or i == len(todo):
            _save_cache(cache)
            logger.info(f"[{i}/{len(todo)}] geocoded, cache saved")

    _save_cache(cache)
    return cache


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _flag_geographic_outliers(out_rows: list[dict]) -> int:
    by_state_district: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in out_rows:
        if row["geocode_status"] == "ok":
            by_state_district[(row["state"], row["district"])].append(row)

    flagged = 0
    for group in by_state_district.values():
        if len(group) < 2:
            continue
        lats = sorted(r["latitude"] for r in group)
        lons = sorted(r["longitude"] for r in group)
        median_lat, median_lon = lats[len(lats) // 2], lons[len(lons) // 2]
        for row in group:
            d = _haversine_km(row["latitude"], row["longitude"], median_lat, median_lon)
            if d > _OUTLIER_DISTANCE_KM:
                row["geocode_status"] = "geo_outlier"
                flagged += 1
    return flagged


def build_reference(output_path: str) -> None:
    triples = _unique_triples()
    logger.info(f"{len(triples)} unique (state, district, market) triples nationwide")

    cache = _load_cache()
    cache = geocode_markets(triples, cache)

    out_rows = []
    for state, district, market in triples:
        key = f"{state}|{district}|{market}"
        geo = cache.get(key, {})
        if "error" in geo:
            out_rows.append({
                "state": state, "district": district, "market": market,
                "latitude": None, "longitude": None, "geocode_status": "unresolved",
            })
        else:
            out_rows.append({
                "state": state, "district": district, "market": market,
                "latitude": geo.get("latitude"), "longitude": geo.get("longitude"),
                "geocode_status": "ok",
            })

    outliers = _flag_geographic_outliers(out_rows)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_rows, f, ensure_ascii=False, indent=2)

    unresolved = sum(1 for r in out_rows if r["geocode_status"] == "unresolved")
    ok = sum(1 for r in out_rows if r["geocode_status"] == "ok")
    logger.info(
        f"Wrote {len(out_rows)} rows to {output_path} "
        f"({ok} geocoded, {unresolved} unresolved, {outliers} flagged as geographic outliers)"
    )


if __name__ == "__main__":
    output_path = sys.argv[1] if len(sys.argv) > 1 else str(DATA_DIR / "nationwide_market_geo_reference.json")
    build_reference(output_path)
