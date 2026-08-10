"""Builds a pincode -> (state, matched Agmarknet district, latitude, longitude)
reference table from a raw postoffice export (e.g. all_ap_postoffices.xlsx —
PIN Code, Post Office, Branch Type, Taluk, District, State, ... columns,
sourced from postalpincode.in).

Why this exists: the postoffice file's District column is the pre-2022
administrative structure (e.g. AP's old 13 districts), while our scraped
Agmarknet data (last_known_prices.json) uses the current structure (AP's 26
districts post-reorg). The two don't line up as a name-to-name crosswalk —
e.g. old "East Godavari" now spans three current districts (East Godavari,
Kakinada, Dr.B.R.A.Konaseema).

Approach: geocode each unique (taluk, state) pair via Nominatim/OSM rather
than the (old_district, state) pair. OSM's own address breakdown
(state_district/county) already reflects CURRENT administrative boundaries,
so a single geocoding call per taluk gives us both lat/long AND the current
district name in one shot -- no separately-sourced crosswalk dataset needed.
Confirmed live: "Donkarayi" (old district "East Godavari" per the postoffice
file) geocodes to state_district="Polavaram", which matches Agmarknet's
actual district list exactly.

Geocoding is per-taluk, not per-pincode: many pincodes share a taluk, and
Nominatim's public instance is rate-limited to ~1 req/sec, so resolving the
~956 unique AP taluks (not the ~1,200 pincodes, and nowhere near the ~10k
post office rows) keeps this to well under 20 minutes.

Some short/ambiguous taluk names can geocode to a same-named place elsewhere
in the state (observed once in testing: "C.k. Palli" matched a place in
Vizianagaram instead of its actual Ananthapur-region location) -- there is no
fully reliable way to catch every one of these automatically, so the output
carries a `match_confidence` field (from the district-name fuzzy match, not
the geocoder itself) and unmatched/low-confidence rows are called out
separately rather than silently trusted.

CLI: python -m bot_processors.location.pincode_geo_reference bot_processors/data/all_ap_postoffices.xlsx
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
from bot_processors.pricing.price_lookup import _fuzzy_match

_CACHE_PATH = DATA_DIR / "pincode_geo_cache.json"
_LAST_KNOWN_PATH = DATA_DIR / "last_known_prices.json"
_NOMINATIM_USER_AGENT = "farm_vaidya_pincode_geocoder"

# The source file's Taluk column often carries an appended administrative
# suffix ("Ainavilli Mandal", "Akividu (mdl)", "Allur Mandalam") that isn't
# part of the actual place name and makes Nominatim return no match at all
# -- confirmed live: stripping these recovered ~half of all first-pass
# geocoding failures. The cache key still uses the raw, unstripped taluk
# name (so it lines up with the source rows unchanged); only the outgoing
# geocode query is cleaned.
_TALUK_SUFFIX_RE = re.compile(
    r"\s*[\(\[]?\s*(mandalam|mandal|manal|mdl|md|nl|rs)\.?\s*[\)\]]?\s*$",
    re.IGNORECASE,
)


def _clean_taluk_for_query(taluk: str) -> str:
    cleaned = _TALUK_SUFFIX_RE.sub("", taluk).strip()
    return cleaned or taluk


# OSM returns a district's official full name (e.g. "Sri Potti Sriramulu
# Nellore"), but Agmarknet's own scraped data uses the acronym form
# ("SPSR Nellore") -- confirmed live: this single alias accounted for 592 of
# 653 "no matching Agmarknet district" rows in the first pass. Plain
# difflib fuzzy matching can't bridge an acronym, so these are listed
# explicitly rather than guessed at.
_DISTRICT_ALIASES = {
    "sri potti sriramulu nellore": "SPSR Nellore",
}


def _resolve_district_alias(geocoded_district: str | None) -> str | None:
    if not geocoded_district:
        return None
    return _DISTRICT_ALIASES.get(geocoded_district.strip().lower())


def _load_cache() -> dict:
    if _CACHE_PATH.exists():
        with open(_CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict) -> None:
    with open(_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _read_postoffice_rows(xlsx_path: str) -> list[dict]:
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb.worksheets[0]
    rows = list(ws.iter_rows(min_row=2, values_only=True))
    out = []
    for r in rows:
        pin, po, btype, taluk, district, state = r[0], r[1], r[2], r[3], r[4], r[5]
        if not pin or not taluk or not state:
            continue
        out.append({
            "pincode": str(pin).strip(),
            "post_office": po,
            "taluk": str(taluk).strip(),
            "old_district": district,
            "state": str(state).strip(),
        })
    return out


def _agmarknet_districts_by_state() -> dict[str, list[str]]:
    """Distinct districts actually seen in our scraped cache, per state --
    this IS Agmarknet's real district vocabulary (there's no separate
    hand-maintained district dropdown; districts only ever come from
    scraped 'All Districts' results), so it's the correct match target."""
    if not _LAST_KNOWN_PATH.exists():
        return {}
    with open(_LAST_KNOWN_PATH, encoding="utf-8") as f:
        data = json.load(f)
    by_state: dict[str, set[str]] = {}
    for entry in data.values():
        state, district = entry.get("state"), entry.get("district")
        if state and district:
            by_state.setdefault(state, set()).add(district)
    return {state: sorted(d) for state, d in by_state.items()}


def geocode_taluks(taluks: list[tuple[str, str]], cache: dict) -> dict:
    """taluks: list of (taluk, state). Mutates and returns cache in place,
    saving to disk after every new geocode so a killed/interrupted run loses
    at most one in-flight request."""
    from geopy.geocoders import Nominatim

    geolocator = Nominatim(user_agent=_NOMINATIM_USER_AGENT)
    todo = [(t, s) for (t, s) in taluks if f"{t}|{s}" not in cache or "error" in cache[f"{t}|{s}"]]
    already_ok = len(taluks) - len(todo)
    logger.info(f"Geocoding {len(todo)} taluks ({already_ok} already resolved and cached)")

    for i, (taluk, state) in enumerate(todo, 1):
        key = f"{taluk}|{state}"
        query = f"{_clean_taluk_for_query(taluk)}, {state}, India"
        try:
            loc = geolocator.geocode(query, addressdetails=True, exactly_one=True, timeout=15)
        except Exception as e:
            logger.warning(f"[{i}/{len(todo)}] geocode failed for {query!r}: {e}")
            cache[key] = {"error": str(e)}
            time.sleep(1)
            continue

        if loc is None:
            cache[key] = {"error": "no result"}
        else:
            addr = loc.raw.get("address", {})
            cache[key] = {
                "latitude": loc.latitude,
                "longitude": loc.longitude,
                "resolved_district": addr.get("state_district") or addr.get("county"),
            }
        if i % 25 == 0 or i == len(todo):
            _save_cache(cache)
            logger.info(f"[{i}/{len(todo)}] geocoded, cache saved")
        time.sleep(1)  # Nominatim public instance usage policy: max 1 req/sec

    _save_cache(cache)
    return cache


_OUTLIER_DISTANCE_KM = 150


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _flag_geographic_outliers(out_rows: list[dict]) -> int:
    """A geocoded taluk can silently resolve to a same-named place elsewhere
    in the state (confirmed live: "Ramachandrapuram" in old district
    "Chittoor" geocoded ~480km away, into "Dr.B.R.A.Konaseema") -- plain
    district-name fuzzy matching can't catch this, since the wrong place can
    still belong to a real, valid Agmarknet district. This flags any "matched"
    row whose coordinates are >150km from the median position of all other
    "matched" rows sharing the same old_district, demoting it to
    "matched_geo_outlier" so it isn't silently trusted alongside the rest.
    Mutates out_rows in place; returns the number of rows flagged."""
    by_old_district: dict[str, list[dict]] = defaultdict(list)
    for row in out_rows:
        if row["match_confidence"] == "matched":
            by_old_district[row["old_district"]].append(row)

    flagged = 0
    for group in by_old_district.values():
        lats = sorted(r["latitude"] for r in group)
        lons = sorted(r["longitude"] for r in group)
        median_lat, median_lon = lats[len(lats) // 2], lons[len(lons) // 2]
        for row in group:
            d = _haversine_km(row["latitude"], row["longitude"], median_lat, median_lon)
            if d > _OUTLIER_DISTANCE_KM:
                row["match_confidence"] = "matched_geo_outlier"
                flagged += 1
    return flagged


def build_reference(xlsx_path: str, output_path: str) -> None:
    rows = _read_postoffice_rows(xlsx_path)
    logger.info(f"Loaded {len(rows)} postoffice rows from {xlsx_path}")

    unique_taluks = sorted({(r["taluk"], r["state"]) for r in rows})
    logger.info(f"{len(unique_taluks)} unique (taluk, state) pairs to geocode")

    cache = _load_cache()
    cache = geocode_taluks(unique_taluks, cache)

    districts_by_state = _agmarknet_districts_by_state()

    out_rows = []
    unresolved = 0
    low_confidence = 0
    for r in rows:
        key = f"{r['taluk']}|{r['state']}"
        geo = cache.get(key, {})
        if "error" in geo:
            unresolved += 1
            out_rows.append({**r, "latitude": None, "longitude": None,
                              "matched_district": None, "match_confidence": "unresolved"})
            continue

        candidates = districts_by_state.get(r["state"], [])
        geocoded_district = geo.get("resolved_district") or ""
        matched = (
            _resolve_district_alias(geocoded_district)
            or (_fuzzy_match(geocoded_district, candidates) if candidates else None)
        )
        confidence = "matched" if matched else "no_agmarknet_district_match"
        if not matched:
            low_confidence += 1

        out_rows.append({
            **r,
            "latitude": geo.get("latitude"),
            "longitude": geo.get("longitude"),
            "geocoded_district": geo.get("resolved_district"),
            "matched_district": matched,
            "match_confidence": confidence,
        })

    outliers = _flag_geographic_outliers(out_rows)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_rows, f, ensure_ascii=False, indent=2)

    logger.info(
        f"Wrote {len(out_rows)} rows to {output_path} "
        f"({unresolved} unresolved, {low_confidence} without a matching Agmarknet district, "
        f"{outliers} flagged as likely geographic mismatches)"
    )


if __name__ == "__main__":
    xlsx_path = sys.argv[1] if len(sys.argv) > 1 else str(DATA_DIR / "all_ap_postoffices.xlsx")
    output_path = sys.argv[2] if len(sys.argv) > 2 else str(DATA_DIR / "pincode_geo_reference.json")
    build_reference(xlsx_path, output_path)
