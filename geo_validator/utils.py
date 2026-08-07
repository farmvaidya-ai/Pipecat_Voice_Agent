"""Shared helpers: logging setup, distance math, coordinate/text cleanup."""

import math
import re
from pathlib import Path
from typing import Optional

from loguru import logger


def setup_logging(log_path: Path) -> None:
    """Routes warnings/errors (retries, API failures, unresolvable rows) to
    `log_path` in addition to stderr, so a run leaves a paper trail without
    cluttering the terminal summary."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(log_path, level="WARNING", encoding="utf-8", mode="w")


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points, in kilometers."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def is_valid_coordinate(lat, lon) -> bool:
    try:
        lat_f, lon_f = float(lat), float(lon)
    except (TypeError, ValueError):
        return False
    if lat_f == 0.0 and lon_f == 0.0:
        return False  # "null island" -- always a bad geocode, never a real Indian location
    return -90.0 <= lat_f <= 90.0 and -180.0 <= lon_f <= 180.0


def clean_pincode(value) -> str:
    """Strips everything but digits — same defensive pattern used live in
    bot_processors/location_lookup.py for caller-dictated pincodes.

    pandas loads a mostly-empty numeric column (like this one -- only
    165/5737 rows have a pincode) as float64, so a real code like 533401
    arrives here as the float 533401.0. str()'ing that gives "533401.0",
    and naively stripping non-digits would keep the "0" after the decimal
    point, corrupting it to "5334010". Collapsing whole-valued floats to
    int first avoids that.
    """
    if isinstance(value, float):
        if value != value:  # NaN
            return ""
        value = int(value)
    return "".join(ch for ch in str(value or "") if ch.isdigit())


def normalize_text(value: Optional[str]) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip().lower()
