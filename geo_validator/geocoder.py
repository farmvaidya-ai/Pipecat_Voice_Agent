"""Reverse geocoding (coordinate -> address) and forward PIN-code geocoding
(PIN code -> its own reference coordinate), backed by the Google Maps
Geocoding API when GOOGLE_MAPS_API_KEY is set, falling back to
Nominatim/OpenStreetMap otherwise. Both paths share one on-disk cache and one
retry/rate-limit wrapper so callers (validator.py) never need to know which
backend is active.

The forward PIN-code lookup exists because a reverse-geocode's own postcode
field only gives a categorical match/mismatch against the input PIN — it
can't say *how far off* a coordinate is. Geocoding the PIN code itself gives
an independent reference point so compare.py can compute a real distance
(see utils.haversine_km).
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import requests
from geopy.geocoders import Nominatim
from loguru import logger

from geo_validator.config import Settings
from geo_validator.utils import is_valid_coordinate


@dataclass
class ReverseGeocodeResult:
    state: str = ""
    district: str = ""
    taluk: str = ""
    village: str = ""
    town: str = ""
    city: str = ""
    suburb: str = ""
    postcode: str = ""
    formatted_address: str = ""

    @property
    def has_data(self) -> bool:
        return bool(self.state or self.district or self.postcode or self.formatted_address)


class GeoCache:
    """Persists reverse/forward geocode results to disk so a duplicate
    coordinate or PIN code (common -- many market yards in the same taluk
    share one geocoded point) is never looked up twice, even across separate
    runs of the validator."""

    def __init__(self, path: Path):
        self._path = path
        self._data: dict = {}
        if path.exists():
            with open(path, encoding="utf-8") as f:
                self._data = json.load(f)

    def get(self, key: str):
        return self._data.get(key)

    def set(self, key: str, value) -> None:
        self._data[key] = value

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)


def _coord_key(lat: float, lon: float) -> str:
    return f"rev:{round(lat, 6)},{round(lon, 6)}"


def _pincode_key(pincode: str, state: str) -> str:
    return f"fwd:{pincode}|{state}"


class ReverseGeocoder:
    def __init__(self, settings: Settings, cache: GeoCache):
        self._settings = settings
        self._cache = cache
        self._nominatim = Nominatim(user_agent=settings.nominatim_user_agent, timeout=settings.request_timeout_s)
        self._last_call = 0.0

    def _throttle(self) -> None:
        rate = self._settings.google_rate_limit_s if self._settings.use_google else self._settings.nominatim_rate_limit_s
        elapsed = time.monotonic() - self._last_call
        if elapsed < rate:
            time.sleep(rate - elapsed)
        self._last_call = time.monotonic()

    def _retrying(self, fn, *args, **kwargs):
        last_exc = None
        for attempt in range(1, self._settings.max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_exc = e
                backoff = self._settings.retry_backoff_base_s * attempt
                logger.warning(f"geocoder: attempt {attempt}/{self._settings.max_retries} failed ({e}); retrying in {backoff:.1f}s")
                time.sleep(backoff)
        logger.error(f"geocoder: all {self._settings.max_retries} attempts failed: {last_exc}")
        return None

    # ---- reverse: coordinate -> address ----

    def reverse(self, lat: float, lon: float) -> ReverseGeocodeResult:
        if not is_valid_coordinate(lat, lon):
            return ReverseGeocodeResult()

        key = _coord_key(lat, lon)
        cached = self._cache.get(key)
        if cached is not None:
            return ReverseGeocodeResult(**cached)

        fn = self._reverse_google if self._settings.use_google else self._reverse_nominatim
        result = self._retrying(fn, lat, lon) or ReverseGeocodeResult()
        self._cache.set(key, asdict(result))
        return result

    def _reverse_google(self, lat: float, lon: float) -> ReverseGeocodeResult:
        self._throttle()
        resp = requests.get(
            "https://maps.googleapis.com/maps/api/geocode/json",
            params={"latlng": f"{lat},{lon}", "key": self._settings.google_maps_api_key},
            timeout=self._settings.request_timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "OK" or not data.get("results"):
            return ReverseGeocodeResult()
        comps = data["results"][0]["address_components"]
        by_type = {}
        for c in comps:
            for t in c["types"]:
                by_type.setdefault(t, c["long_name"])
        return ReverseGeocodeResult(
            state=by_type.get("administrative_area_level_1", ""),
            district=by_type.get("administrative_area_level_2", ""),
            taluk=by_type.get("administrative_area_level_3", ""),
            village=by_type.get("sublocality", "") or by_type.get("neighborhood", ""),
            town=by_type.get("postal_town", ""),
            city=by_type.get("locality", ""),
            suburb=by_type.get("sublocality_level_1", ""),
            postcode=by_type.get("postal_code", ""),
            formatted_address=data["results"][0].get("formatted_address", ""),
        )

    def _reverse_nominatim(self, lat: float, lon: float) -> ReverseGeocodeResult:
        self._throttle()
        loc = self._nominatim.reverse((lat, lon), addressdetails=True, zoom=18, exactly_one=True)
        if loc is None:
            return ReverseGeocodeResult()
        addr = loc.raw.get("address", {})
        return ReverseGeocodeResult(
            state=addr.get("state", ""),
            district=addr.get("state_district") or addr.get("county", ""),
            taluk=addr.get("county", ""),
            village=addr.get("village", ""),
            town=addr.get("town", ""),
            city=addr.get("city", ""),
            suburb=addr.get("suburb", ""),
            postcode=addr.get("postcode", ""),
            formatted_address=loc.raw.get("display_name", ""),
        )

    # ---- forward: PIN code -> its own reference coordinate ----

    def geocode_pincode(self, pincode: str, state: str) -> Optional[tuple]:
        if not pincode:
            return None
        key = _pincode_key(pincode, state)
        cached = self._cache.get(key)
        if cached is not None:
            return tuple(cached) if cached else None

        fn = self._geocode_pincode_google if self._settings.use_google else self._geocode_pincode_nominatim
        coords = self._retrying(fn, pincode, state)
        self._cache.set(key, list(coords) if coords else None)
        return coords

    def _geocode_pincode_google(self, pincode: str, state: str) -> Optional[tuple]:
        self._throttle()
        resp = requests.get(
            "https://maps.googleapis.com/maps/api/geocode/json",
            params={"address": f"{pincode}, {state}, India", "key": self._settings.google_maps_api_key},
            timeout=self._settings.request_timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "OK" or not data.get("results"):
            return None
        loc = data["results"][0]["geometry"]["location"]
        return (loc["lat"], loc["lng"])

    def _geocode_pincode_nominatim(self, pincode: str, state: str) -> Optional[tuple]:
        self._throttle()
        loc = self._nominatim.geocode({"postalcode": pincode, "country": "India", "state": state}, exactly_one=True)
        if loc is None:
            self._throttle()
            loc = self._nominatim.geocode(f"{pincode}, {state}, India", exactly_one=True)
        return (loc.latitude, loc.longitude) if loc else None
