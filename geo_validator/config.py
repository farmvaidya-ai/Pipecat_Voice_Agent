"""Configuration for the market-yard geo-coordinate validator: API keys,
rate limits, matching thresholds, and file paths — all overridable via
environment variables (.env) or CLI flags, nothing hardcoded per-machine.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

_HERE = Path(__file__).parent


@dataclass(frozen=True)
class Settings:
    google_maps_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_MAPS_API_KEY", ""))
    nominatim_user_agent: str = "farmvaidya_geo_validator"

    request_timeout_s: float = 15.0
    max_retries: int = 3
    retry_backoff_base_s: float = 2.0

    # OSM's public Nominatim instance forbids more than ~1 req/sec; Google's
    # Geocoding API default quota is far higher (50 req/sec) but this stays
    # conservative since it's configurable, not something to tune blind.
    nominatim_rate_limit_s: float = 1.0
    google_rate_limit_s: float = 0.05

    district_fuzzy_threshold: float = 90.0
    accurate_distance_km: float = 2.0
    nearby_distance_km: float = 10.0

    cache_path: Path = _HERE / "output" / "geocode_cache.json"
    default_output_path: Path = _HERE / "output" / "Verified_Market_Geo.xlsx"
    default_log_path: Path = _HERE / "output" / "validation.log"

    @property
    def use_google(self) -> bool:
        return bool(self.google_maps_api_key)


SETTINGS = Settings()


# Confirmed, non-ambiguous Andhra Pradesh district name variants, all mapped
# to one canonical spelling — deliberately NOT a full 13-to-26 crosswalk.
# Most of the 2022 AP reorg *split* one old district into several new ones
# (e.g. old East Godavari now spans East Godavari, Kakinada, and
# Dr.B.R.Ambedkar Konaseema), so a single alias would be actively wrong for
# most rows in a split district. Only entries that are a genuine 1:1 rename
# or spelling variant with no split belong here — everything else is left to
# fuzzy matching + distance, so a real mismatch still surfaces instead of
# being silently forced to "match".
#
# Includes both directions of "abbreviated official name" <-> "OSM's spelled
# -out name" for the same district (e.g. Agmarknet says "SPSR Nellore", OSM's
# reverse-geocode says "Sri Potti Sriramulu Nellore") -- confirmed live via
# this tool's own smoke test: that pair scored only 56% on token_sort_ratio,
# well under the 90% threshold, and was wrongly flagged as a mismatch before
# this alias was added.
AP_DISTRICT_RENAMES: dict[str, str] = {
    "cuddapah": "YSR Kadapa",
    "kadapa": "YSR Kadapa",
    "y.s.r.": "YSR Kadapa",
    "ysr district": "YSR Kadapa",
    "nellore": "SPSR Nellore",
    "sri potti sriramulu nellore": "SPSR Nellore",
}
