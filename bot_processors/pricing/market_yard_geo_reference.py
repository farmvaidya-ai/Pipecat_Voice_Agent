"""Builds a (state, district, market yard) -> (pincode, latitude, longitude)
reference table from bot_processors/AP_Telangana_Market_Yards_filled.xlsx
(State, District, Market Yard, Pincodes columns).

Unlike pincode_geo_reference.py (built from postoffice data, needing OSM's
own district field to bridge the pre/post-2022 AP district reorg), this
source file already lists CURRENT district names directly -- confirmed live:
92 of 96 markets in last_known_prices.json (AP + Telangana) have an exact
name match here. So there's no district-matching step at all: geocoding is
only used to add latitude/longitude: the state/district/pincode come
straight from the sheet.

Market yard names geocode far more reliably than the postoffice file's raw
taluk names (they're well-known town names), but carry the same suffix-noise
problem -- confirmed live: "Ravulapalem (Kothapeta APMC)" and
"Vanasthalipuram,RBZ" return NO RESULT until the parenthetical/comma suffix
is stripped.

CLI: python -m bot_processors.pricing.market_yard_geo_reference
"""

import json
import math
import re
import sys
import time
from collections import defaultdict

import openpyxl
from loguru import logger

from bot_processors.paths import DATA_DIR

_SOURCE_PATH = DATA_DIR / "AP_Telangana_Market_Yards_filled.xlsx"
_CACHE_PATH = DATA_DIR / "market_yard_geo_cache.json"
_NOMINATIM_USER_AGENT = "farm_vaidya_pincode_geocoder"
_OUTLIER_DISTANCE_KM = 150

# Strip parenthetical alt-names ("(Kothapeta APMC)"), trailing comma-attached
# codes (",RBZ"), and the generic "APMC"/"Market"/"Mandi" market-type suffix
# -- confirmed live: all three break Nominatim matching for otherwise-valid
# town names.
_PAREN_RE = re.compile(r"\([^)]*\)")
_MARKET_SUFFIX_RE = re.compile(r"\s*\b(apmc|market|mandi|rbz)\b\.?\s*$", re.IGNORECASE)


def _clean_market_for_query(market: str) -> str:
    cleaned = _PAREN_RE.sub("", market)
    cleaned = cleaned.split(",")[0]
    cleaned = _MARKET_SUFFIX_RE.sub("", cleaned).strip()
    return cleaned or market


def _read_market_rows() -> list[dict]:
    wb = openpyxl.load_workbook(_SOURCE_PATH, read_only=True, data_only=True)
    ws = wb.worksheets[0]
    out = []
    for r in ws.iter_rows(min_row=2, values_only=True):
        state, district, market, pincode = r[0], r[1], r[2], r[3]
        if not state or not market:
            continue
        pincode_str = str(pincode).strip()
        out.append({
            "state": str(state).strip(),
            "district": str(district).strip() if district else None,
            "market": str(market).strip(),
            "pincode": pincode_str if pincode_str.lower() != "not found" else None,
        })
    return out


def _load_cache() -> dict:
    if _CACHE_PATH.exists():
        with open(_CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict) -> None:
    with open(_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def geocode_markets(rows: list[dict], cache: dict) -> dict:
    """Keyed by 'state|district|market'. Tries '{market}, {district}, {state},
    India' first (district disambiguates common names, e.g. Koratla ->
    Jagtial not Adilabad), falling back to '{market}, {state}, India' since
    some current district names aren't themselves recognized by Nominatim
    (e.g. "Polavaram" as a search term, even though it's a real district)."""
    from geopy.geocoders import Nominatim

    geolocator = Nominatim(user_agent=_NOMINATIM_USER_AGENT)
    keys = [(r["state"], r["district"], r["market"]) for r in rows]
    todo = [k for k in keys if f"{k[0]}|{k[1]}|{k[2]}" not in cache or "error" in cache[f"{k[0]}|{k[1]}|{k[2]}"]]
    already_ok = len(keys) - len(todo)
    logger.info(f"Geocoding {len(todo)} market yards ({already_ok} already resolved and cached)")

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
    """District here comes straight from the source sheet (already current),
    not derived from geocoding, so any row >150km from its district's other
    markets is purely a geocoding mistake -- e.g. Nominatim picking a
    same-named place in another state."""
    by_district: dict[str, list[dict]] = defaultdict(list)
    for row in out_rows:
        if row["geocode_status"] == "ok":
            by_district[row["district"]].append(row)

    flagged = 0
    for group in by_district.values():
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
    rows = _read_market_rows()
    logger.info(f"Loaded {len(rows)} market yard rows from {_SOURCE_PATH.name}")

    cache = _load_cache()
    cache = geocode_markets(rows, cache)

    out_rows = []
    for r in rows:
        key = f"{r['state']}|{r['district']}|{r['market']}"
        geo = cache.get(key, {})
        if "error" in geo:
            out_rows.append({**r, "latitude": None, "longitude": None, "geocode_status": "unresolved"})
        else:
            out_rows.append({**r, "latitude": geo.get("latitude"), "longitude": geo.get("longitude"),
                              "geocode_status": "ok"})

    outliers = _flag_geographic_outliers(out_rows)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_rows, f, ensure_ascii=False, indent=2)

    unresolved = sum(1 for r in out_rows if r["geocode_status"] == "unresolved")
    ok = sum(1 for r in out_rows if r["geocode_status"] == "ok")
    no_pincode = sum(1 for r in out_rows if r["pincode"] is None)
    logger.info(
        f"Wrote {len(out_rows)} rows to {output_path} "
        f"({ok} geocoded, {unresolved} unresolved, {outliers} flagged as geographic outliers, "
        f"{no_pincode} missing pincode in source sheet)"
    )


if __name__ == "__main__":
    output_path = sys.argv[1] if len(sys.argv) > 1 else str(DATA_DIR / "market_yard_geo_reference.json")
    build_reference(output_path)
