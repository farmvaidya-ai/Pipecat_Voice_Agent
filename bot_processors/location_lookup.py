"""Resolves a caller's spoken location + pincode into coordinates, then ranks
Agmarknet districts by proximity so the bot can read out nearby candidates for
the caller to confirm — see system_prompt.txt's "Caller location" section.
Design: ask the caller's spoken location (village/mandal/district) first,
pincode second, then confirm from a geography-ranked candidate list (however
many districts are genuinely nearby, not a fixed count) rather than the bot
guessing a single district and asking yes/no.

Two-tier coordinate lookup, in order:
  1. OpenWeatherMap's ZIP endpoint resolves the caller's pincode directly to
     coordinates, nationwide — used in preference to the AP-only
     postoffice-based pincode_geo_reference.json, which only ever covers
     Andhra Pradesh.
  2. If the pincode doesn't resolve (typo, or OpenWeatherMap has no ZIP data
     for it), falls back to geocoding the caller's spoken place name instead,
     reusing weather_lookup._geocode (the same OpenWeatherMap Direct
     Geocoding call get_weather already relies on).

Nearby-district ranking reuses the district centroids computed from
nationwide_market_geo_reference.json / market_yard_geo_reference.json (mean
lat/lon of every geocoded market in that (state, district)) — the same
reference data enrich_prices_with_geo.py uses to enrich last_known_prices.json.

CLI: none — these are real LLM tool calls, registered on the LLMContext's
`tools=[...]` list in Bot.py (see make_confirm_location/make_save_caller_location).
"""

import math
import os
import time
from pathlib import Path

import json
import requests
from loguru import logger

from pipecat.services.llm_service import FunctionCallParams

from bot_processors.call_db import log_tool_call
from bot_processors.caller_db import save_location
from bot_processors.price_lookup import _resolve_state
from bot_processors.task_tracker import track_task
from bot_processors.weather_lookup import _geocode as _geocode_place

_OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY", "")

_MARKET_GEO_PATH = Path(__file__).parent / "market_yard_geo_reference.json"
_NATIONWIDE_GEO_PATH = Path(__file__).parent / "nationwide_market_geo_reference.json"

# "Surrounding/neighboring" districts, not a fixed count: everything within
# the radius, capped so the bot doesn't read out an unreasonably long list;
# if fewer than the minimum fall inside the radius (sparse geocoding in that
# area), the nearest few are offered anyway rather than returning nothing.
_NEARBY_RADIUS_KM = 60
_NEARBY_MAX_CANDIDATES = 8
_NEARBY_MIN_CANDIDATES = 3


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


_centroids: dict[tuple[str, str], tuple[float, float]] = {}
_centroids_mtimes: tuple[float, float] = (0.0, 0.0)


def _load_centroids() -> dict[tuple[str, str], tuple[float, float]]:
    points: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for path in (_NATIONWIDE_GEO_PATH, _MARKET_GEO_PATH):
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)
        for r in rows:
            if r.get("geocode_status") != "ok" or r.get("latitude") is None:
                continue
            key = (r["state"], r["district"])
            points.setdefault(key, []).append((r["latitude"], r["longitude"]))
    return {
        key: (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
        for key, pts in points.items()
    }


def _reload_centroids_if_changed() -> None:
    """Same mtime-guarded reload pattern as price_lookup._reload_last_known_if_changed
    — these reference files are rebuilt by the offline geocoding scripts, not
    by this long-running bot process."""
    global _centroids, _centroids_mtimes
    m1 = _NATIONWIDE_GEO_PATH.stat().st_mtime if _NATIONWIDE_GEO_PATH.exists() else 0.0
    m2 = _MARKET_GEO_PATH.stat().st_mtime if _MARKET_GEO_PATH.exists() else 0.0
    if (m1, m2) == _centroids_mtimes:
        return
    _centroids = _load_centroids()
    _centroids_mtimes = (m1, m2)


def _nearby_districts(state: str, lat: float, lon: float) -> list[tuple[str, float]]:
    """Districts of `state` sorted by distance from (lat, lon), nearest
    first — everything within _NEARBY_RADIUS_KM, or the closest few if the
    radius catches too few (sparse geocoding coverage in that area)."""
    _reload_centroids_if_changed()
    same_state = sorted(
        (
            (district, _haversine_km(lat, lon, c_lat, c_lon))
            for (s, district), (c_lat, c_lon) in _centroids.items()
            if s == state
        ),
        key=lambda x: x[1],
    )
    within_radius = [d for d in same_state if d[1] <= _NEARBY_RADIUS_KM]
    if len(within_radius) >= _NEARBY_MIN_CANDIDATES:
        return within_radius[:_NEARBY_MAX_CANDIDATES]
    return same_state[:_NEARBY_MIN_CANDIDATES]


def _geocode_pincode(pincode: str) -> tuple[float, float] | None:
    if not _OPENWEATHERMAP_API_KEY or not pincode:
        return None
    try:
        resp = requests.get(
            "https://api.openweathermap.org/geo/1.0/zip",
            params={"zip": f"{pincode.strip()},IN", "appid": _OPENWEATHERMAP_API_KEY},
            timeout=5,
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        return data["lat"], data["lon"]
    except Exception as e:
        logger.info(f"location_lookup: pincode geocode failed for {pincode!r}: {e}")
        return None


def make_confirm_location(serializer=None):
    """Builds the confirm_location direct-function tool — same
    serializer-closure pattern as make_get_price/make_get_weather, so tool
    outcomes are logged with the right call_id."""

    async def confirm_location(params: FunctionCallParams, pincode: str, place: str, state: str) -> None:
        """Resolve the caller's location into a short list of nearby
        Agmarknet districts for them to confirm.

        Call this only after the caller has told you BOTH their spoken
        location (village, town, or mandal/district name) AND their 6-digit
        pincode — never call it with just one of the two, and never call it
        if the caller's location is already known from a system note. This
        does not save anything yet; it only returns candidate district names
        for you to read aloud and ask the caller which one is theirs. Once
        they confirm one by name, call save_caller_location with that exact
        district.

        Args:
            pincode: The 6-digit postal pincode the caller said.
            place: The village, town, or mandal/district name the caller said.
            state: The Indian state or union territory the caller is calling from.
        """
        _t0 = time.monotonic()

        resolved_state = _resolve_state(state)
        if resolved_state is None:
            await params.result_callback({
                "error": f"{state!r} isn't a recognized state/UT — ask the caller to confirm which state.",
            })
            return

        coords = _geocode_pincode(pincode)
        resolved_from = "pincode"
        if coords is None:
            coords = _geocode_place(f"{place}, {resolved_state}")
            resolved_from = "place name"
        if coords is None:
            await params.result_callback({
                "error": "Could not resolve this location — ask the caller to repeat their village/town and pincode.",
            })
            return

        candidates = _nearby_districts(resolved_state, *coords)
        exec_ms = int((time.monotonic() - _t0) * 1000)

        if not candidates:
            await params.result_callback({
                "error": f"No known Agmarknet districts found near this location in {resolved_state}.",
            })
        else:
            await params.result_callback({
                "state": resolved_state,
                "resolved_from": resolved_from,
                "candidate_districts": [d for d, _ in candidates],
            })

        if serializer is not None and serializer.call_id:
            outcome = ", ".join(d for d, _ in candidates) if candidates else "no match"
            track_task(log_tool_call(
                serializer.call_id, "confirm_location",
                f"pincode={pincode!r} place={place!r} state={state!r}", outcome, bool(candidates), exec_ms,
            ))

    return confirm_location


def make_save_caller_location(serializer=None):
    """Builds the save_caller_location direct-function tool."""

    async def save_caller_location(params: FunctionCallParams, district: str, state: str, pincode: str) -> None:
        """Save the caller's confirmed farming district so future calls from
        this phone number don't need to ask for their location again.

        Call this only after the caller has verbally confirmed which
        district (from the list confirm_location returned) is theirs.

        Args:
            district: The exact district name the caller confirmed, as
                returned by confirm_location's candidate_districts.
            state: The state/UT this district is in.
            pincode: The pincode the caller gave earlier in this call.
        """
        _t0 = time.monotonic()
        phone_number = serializer.caller_number if serializer else ""
        resolved_state = _resolve_state(state) or state

        ok = await save_location(phone_number, resolved_state, district, pincode)
        exec_ms = int((time.monotonic() - _t0) * 1000)

        if ok:
            await params.result_callback({"saved": True, "district": district, "state": resolved_state})
        else:
            await params.result_callback({"error": "Could not save the location right now."})

        if serializer is not None and serializer.call_id:
            track_task(log_tool_call(
                serializer.call_id, "save_caller_location",
                f"{district}, {resolved_state} ({pincode})", "saved" if ok else "failed", ok, exec_ms,
            ))

    return save_caller_location
