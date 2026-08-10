"""Weather lookup: a real LLM tool call. The LLM extracts whatever place
name the caller said (or asks for one if they didn't say it) and calls
get_weather(location=...) directly — no location is ever hardcoded here or
resolved via a fixed alias table.

Unlike bot_processors.pricing.price_lookup, this fetches live, per-call — both IMD
and OpenWeatherMap are direct JSON APIs (no browser, no cascading form), so
a lookup is a sub-second network round trip rather than the several seconds
a single agmarknet.gov.in commodity scrape takes. No background job, no
cache file.

Unlike RAGInjector (bot_processors/rag/rag.py) and PriceLookupInjector
(bot_processors/pricing/price_lookup.py), this is not a FrameProcessor that injects
a synthetic system message ahead of every turn — it's a genuine pipecat
"direct function" tool (see pipecat.adapters.schemas.direct_function),
registered on the LLMContext's `tools=[...]` list in Bot.py. The LLM decides
on its own, per turn, whether the caller's words justify calling it, and
with what argument; there is no keyword/intent pre-detection step here at
all, so there's nothing that needs a district dictionary to resolve against.

Two-tier source: IMD (India Meteorological Department, api.imd.gov.in) is
tried first when IMD_API_KEY is set, since it's the official government
source for Indian weather; OpenWeatherMap is always the fallback (used when
IMD_API_KEY is unset, or the IMD call fails, or IMD has no station near the
caller's location). Both tiers return the same dict shape so fetch_weather()
and get_weather() don't need to know which source answered.

IMD caveats (as of 2026-07-30, api.imd.gov.in/public/api_reference.html):
the official docs don't publish a sample JSON response for /current_wx or
/cityforecast_mapping, and the weather-code table is only documented as
coarse ranges (e.g. 60-69 = rain variants), not an exact code→text mapping.
_extract() below tolerates several plausible key-casing variants per field
so a first real call (once IMD_API_KEY is set) is more likely to parse
without a code change; _IMD_WEATHER_CODE_RANGES is a best-effort
approximation of the documented ranges, not a verified exact table. Treat
the first real response as the thing to check the field names/codes
against, not these guesses.
"""

import asyncio
import os
import threading
import time
from datetime import datetime, timezone

import requests
from loguru import logger

from pipecat.services.llm_service import FunctionCallParams

from bot_processors.calls.call_db import log_tool_call
from bot_processors.core.task_tracker import track_task

_OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY", "")
_IMD_API_KEY = os.getenv("IMD_API_KEY", "")
_IMD_BASE_URL = os.getenv("IMD_BASE_URL", "https://api.imd.gov.in/api/v1").rstrip("/")


# ──────────────────────────────────────────────────────────────────────────────
# IMD (India Meteorological Department) — primary tier
# ──────────────────────────────────────────────────────────────────────────────

def _extract(data: dict, *aliases: str):
    """Case-insensitive, alias-tolerant field lookup. IMD's docs don't publish
    a sample JSON response, so the exact key casing (Station_Id vs StationId
    vs station_id) is unverified — this tries every plausible variant instead
    of hardcoding one and silently returning None on a mismatch."""
    lowered = {k.lower().replace(" ", "").replace("_", ""): v for k, v in data.items()}
    for alias in aliases:
        key = alias.lower().replace(" ", "").replace("_", "")
        if key in lowered:
            return lowered[key]
    return None


# Coarse approximation of IMD's documented weather-code ranges (their own
# reference page only lists ranges like "60-69 = rain variants", not an
# exact 01-99 table) — good enough to pick a Telugu-sayable description,
# but verify against a real response once IMD_API_KEY is live.
_IMD_WEATHER_CODE_RANGES: list[tuple[range, str]] = [
    (range(0, 4), "clear or partly cloudy"),
    (range(4, 10), "dusty/hazy"),
    (range(10, 13), "misty/foggy"),
    (range(13, 20), "thunderstorm or squall nearby"),
    (range(20, 28), "drizzle or light rain recently"),
    (range(28, 29), "fog"),
    (range(29, 30), "thunderstorm"),
    (range(30, 36), "dust storm or sand storm"),
    (range(36, 40), "blowing/drifting snow"),
    (range(40, 50), "fog"),
    (range(50, 60), "drizzle"),
    (range(60, 70), "rain"),
    (range(70, 80), "snow or ice"),
    (range(80, 91), "rain showers"),
    (range(91, 100), "thunderstorm with rain"),
]


def _imd_weather_code_description(code) -> str:
    try:
        code_int = int(code)
    except (TypeError, ValueError):
        return "weather condition unavailable"
    for code_range, description in _IMD_WEATHER_CODE_RANGES:
        if code_int in code_range:
            return description
    return "weather condition unavailable"


# IMD identifies locations by station ID, not free-text place names — the
# mapping endpoint below is fetched once and cached (not hardcoded per the
# module docstring's no-alias-table rule: this is the live list IMD itself
# publishes, refreshed daily, not a table we authored).
_imd_station_lock = threading.Lock()
_imd_stations: list[dict] | None = None
_imd_stations_fetched_at: float = 0.0
_IMD_STATION_CACHE_SECS = 24 * 60 * 60


def _imd_headers() -> dict:
    # IMD's docs describe "Secure JWT" auth but don't show the exact header
    # name/format — Authorization: Bearer is the common convention, but if
    # the first real call 401s, this is the first thing to check.
    return {"Authorization": f"Bearer {_IMD_API_KEY}"}


def _get_imd_stations() -> list[dict]:
    global _imd_stations, _imd_stations_fetched_at
    with _imd_station_lock:
        if _imd_stations is not None and (time.monotonic() - _imd_stations_fetched_at) < _IMD_STATION_CACHE_SECS:
            return _imd_stations
        resp = requests.get(
            f"{_IMD_BASE_URL}/cityforecast_mapping",
            headers=_imd_headers(),
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        stations = data if isinstance(data, list) else data.get("data", data.get("results", []))
        logger.debug(f"weather_lookup(IMD): fetched {len(stations)} stations, sample={stations[:1]}")
        _imd_stations = stations
        _imd_stations_fetched_at = time.monotonic()
        return _imd_stations


def _resolve_imd_station(location: str) -> str | None:
    """Fuzzy-matches the caller's free-text location against IMD's own
    station name list. Returns the station id, or None if nothing close
    enough is found (caller falls back to OpenWeatherMap in that case)."""
    import difflib

    stations = _get_imd_stations()
    names = []
    name_to_id = {}
    for station in stations:
        name = _extract(station, "Station Name", "StationName", "City", "District", "Name")
        station_id = _extract(station, "Station Id", "StationId", "Id")
        if name and station_id is not None:
            names.append(name)
            name_to_id[name] = station_id

    matches = difflib.get_close_matches(location, names, n=1, cutoff=0.6)
    if not matches:
        return None
    return name_to_id[matches[0]]


def _fetch_imd(location: str) -> dict | None:
    if not _IMD_API_KEY:
        return None
    try:
        station_id = _resolve_imd_station(location)
        if station_id is None:
            logger.info(f"weather_lookup(IMD): no station match for {location!r}, falling back to OpenWeatherMap")
            return None

        resp = requests.get(
            f"{_IMD_BASE_URL}/current_wx",
            params={"id": station_id},
            headers=_imd_headers(),
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            data = data[0] if data else {}

        temperature_c = _extract(data, "Temperature", "Temp")
        humidity_pct = _extract(data, "Humidity", "RH")
        wind_kmph = _extract(data, "Wind Speed", "WindSpeed")
        rainfall_mm = _extract(data, "Last 24 hrs Rainfall", "Rainfall 24 Hrs", "Rainfall")
        weather_code = _extract(data, "Weather Code", "WeatherCode", "Weather")

        if temperature_c is None:
            logger.warning(f"weather_lookup(IMD): unrecognised response shape for {location!r}: {data}")
            return None

        return {
            "source": "IMD",
            "temperature_c": float(temperature_c),
            "humidity_pct": float(humidity_pct) if humidity_pct is not None else None,
            # IMD's current_wx exposes a 24h rainfall total, not a 1h figure
            # like OpenWeatherMap's rain.1h — closest available field.
            "precipitation_mm": float(rainfall_mm) if rainfall_mm is not None else 0.0,
            # Already KMPH per IMD's docs, unlike OpenWeatherMap's m/s — no
            # unit conversion here (do NOT reapply the ×3.6 used below).
            "wind_kmh": float(wind_kmph) if wind_kmph is not None else None,
            "description": _imd_weather_code_description(weather_code),
            "as_of": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.info(f"weather_lookup(IMD): lookup failed for {location!r} ({e}), falling back to OpenWeatherMap")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# OpenWeatherMap — fallback tier
# ──────────────────────────────────────────────────────────────────────────────

def _weather_by_coords(lat: float, lon: float) -> dict:
    resp = requests.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={"lat": lat, "lon": lon, "appid": _OPENWEATHERMAP_API_KEY, "units": "metric"},
        timeout=5,
    )
    resp.raise_for_status()
    data = resp.json()
    return {
        "source": "OpenWeatherMap",
        "temperature_c": data["main"]["temp"],
        "humidity_pct": data["main"]["humidity"],
        "precipitation_mm": data.get("rain", {}).get("1h", 0.0),
        "wind_kmh": round(data["wind"]["speed"] * 3.6, 1),
        "description": data["weather"][0]["description"],
        "as_of": datetime.now(timezone.utc).isoformat(),
    }


def _geocode(location: str) -> tuple[float, float] | None:
    """Falls back to OpenWeatherMap's own Geocoding API, which does fuzzy
    name matching against a much broader gazetteer than the direct-weather
    endpoint's `q` param — the direct endpoint 404s on plenty of small
    Indian villages/mandals that this API still resolves. Returns the first
    (best) match's coordinates, or None if it has nothing either."""
    resp = requests.get(
        "https://api.openweathermap.org/geo/1.0/direct",
        params={"q": f"{location},IN", "limit": 1, "appid": _OPENWEATHERMAP_API_KEY},
        timeout=5,
    )
    resp.raise_for_status()
    matches = resp.json()
    if not matches:
        return None
    return matches[0]["lat"], matches[0]["lon"]


def _fetch_openweathermap(location: str) -> dict | None:
    """OpenWeatherMap's `q` param already accepts any free-text place name
    (city, village, town) worldwide — this is why no fixed location table is
    needed anywhere in this refactor. `,IN` is appended to bias results
    toward India, since this bot's callers are all Indian farmers and a bare
    place name like "Guntur" could otherwise match a same-named place
    elsewhere.

    A direct name query still 404s on plenty of small villages/mandals that
    aren't in the weather endpoint's own place database (flagged live
    2026-07-29 by a tester: "Weather information is unavailable for a few
    locations"). Falling back to the Geocoding API's fuzzier gazetteer, then
    querying weather by coordinates, recovers most of those without hardcoding
    any location."""
    if not _OPENWEATHERMAP_API_KEY:
        return None
    try:
        resp = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"q": f"{location},IN", "appid": _OPENWEATHERMAP_API_KEY, "units": "metric"},
            timeout=5,
        )
        if resp.status_code == 404:
            raise requests.HTTPError(response=resp)
        resp.raise_for_status()
        data = resp.json()
        return {
            "source": "OpenWeatherMap",
            "temperature_c": data["main"]["temp"],
            "humidity_pct": data["main"]["humidity"],
            "precipitation_mm": data.get("rain", {}).get("1h", 0.0),
            "wind_kmh": round(data["wind"]["speed"] * 3.6, 1),
            "description": data["weather"][0]["description"],
            "as_of": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.info(f"weather_lookup: direct name lookup failed for {location!r} ({e}), trying geocoding fallback")
        try:
            coords = _geocode(location)
            if coords is None:
                logger.warning(f"⚠️ weather_lookup: no data at all (direct or geocoded) for {location!r}")
                return None
            return _weather_by_coords(*coords)
        except Exception as e2:
            logger.warning(f"⚠️ weather_lookup: geocoding fallback also failed for {location!r}: {e2}")
            return None


async def fetch_weather(location: str) -> dict | None:
    """requests is a synchronous library, run off the event loop via
    asyncio.to_thread — same pattern as bot_processors/calls/escalation.py's
    outbound HTTP call.

    Tier order is "auto" (IMD, then OpenWeatherMap) by default — this is a
    fallback chain, not a provider switch like LLM_PROVIDER, because a
    farmer on a live call should still get an answer if IMD is briefly down
    or doesn't cover their village, not a config someone has to flip back.

    WEATHER_PROVIDER is a debug-only override (not meant for production use):
    with auto-fallback on, an IMD field-parsing bug would be silently masked
    by OpenWeatherMap answering instead — set WEATHER_PROVIDER=imd while
    testing to force IMD-only (returns None, no fallback, on any failure) and
    confirm it's actually parsing before trusting the default "auto" tier.
    """
    mode = os.getenv("WEATHER_PROVIDER", "auto").lower().strip()

    if mode == "owm":
        return await asyncio.to_thread(_fetch_openweathermap, location)

    result = await asyncio.to_thread(_fetch_imd, location)
    if result or mode == "imd":
        return result
    return await asyncio.to_thread(_fetch_openweathermap, location)


def make_get_weather(serializer=None):
    """Builds the get_weather direct-function tool, closing over this call's
    serializer so tool-call outcomes can still be logged via log_tool_call
    (bot_processors/calls/call_db.py) — Bot.py builds one LLMContext/serializer per
    call, so each call gets its own get_weather closure with the right
    call_id baked in. The LLM only ever sees the plain get_weather(location)
    signature below; the closure is invisible to it.
    """

    async def get_weather(params: FunctionCallParams, location: str) -> None:
        """Get the current weather for a place the caller named.

        Call this whenever the caller asks about weather, rain, temperature,
        humidity, wind, or storms. Only call this once you know a specific
        city, village, town, or district — if the caller hasn't said one yet,
        do not call this function; ask them which city, village, or location
        they mean first instead.

        Args:
            location: The city, village, town, or district the caller said,
                as close to their own words as possible (for example
                "Hyderabad", "Warangal", "Ongole"). Never guess or invent a
                location that the caller didn't say.
        """
        _t0 = time.monotonic()
        result = await fetch_weather(location)
        exec_ms = int((time.monotonic() - _t0) * 1000)

        if result:
            await params.result_callback({
                "location": location,
                "source": result["source"],
                "description": result["description"],
                "temperature_c": result["temperature_c"],
                "humidity_pct": result["humidity_pct"],
                "wind_kmh": result["wind_kmh"],
                "as_of": result["as_of"],
            })
        else:
            await params.result_callback({
                "error": (
                    f"No weather data could be fetched for {location!r} right now — "
                    "this is a live-lookup failure, not a place that doesn't exist."
                ),
            })

        if serializer is not None and serializer.call_id:
            outcome = (
                f"{result['description']}, {result['temperature_c']:.1f}°C"
                if result else "no data found"
            )
            track_task(log_tool_call(
                serializer.call_id, "weather_lookup",
                f"weather in {location}", outcome, bool(result), exec_ms,
            ))

    return get_weather
