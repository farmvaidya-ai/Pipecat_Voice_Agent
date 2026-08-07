"""Field-by-field comparison between an input row's stated location and its
reverse-geocoded location: state/district/PIN matching (with AP district
alias + fuzzy matching), status classification, and confidence scoring.

Status precedence (most to least certain), since the user-specified rules
can overlap on a single row:
  1. Manual Review -- reverse geocoder returned nothing usable at all.
  2. Accurate -- PIN, district, and state all match, and the coordinate is
     within `accurate_distance_km` of the PIN's own reference point.
  3. Nearby -- district matches and the distance is between the accurate and
     nearby thresholds (close, but not tight enough to call exact).
  4. Incorrect -- everything else, including the case where every field
     matches but the coordinate is still implausibly far from the PIN's own
     reference point (a PIN code covers a small area, so that distance alone
     is still a real red flag).
"""

from dataclasses import dataclass
from typing import Optional

from rapidfuzz import fuzz

from geo_validator.config import AP_DISTRICT_RENAMES, Settings
from geo_validator.utils import clean_pincode, normalize_text


@dataclass
class ComparisonResult:
    state_match: bool
    district_match: bool
    pin_match: bool
    distance_km: Optional[float]
    status: str
    confidence: str
    remarks: str


def _normalize_district(name: str) -> str:
    norm = normalize_text(name)
    return normalize_text(AP_DISTRICT_RENAMES.get(norm, name))


def district_similarity(input_district: str, geocoded_district: str) -> float:
    a, b = _normalize_district(input_district), _normalize_district(geocoded_district)
    if not a or not b:
        return 0.0
    return fuzz.token_sort_ratio(a, b)


def compare_row(
    input_state: str,
    input_district: str,
    input_pincode: str,
    geocoded_state: str,
    geocoded_district: str,
    geocoded_pincode: str,
    has_geocode_data: bool,
    distance_km: Optional[float],
    settings: Settings,
) -> ComparisonResult:
    if not has_geocode_data:
        return ComparisonResult(
            state_match=False,
            district_match=False,
            pin_match=False,
            distance_km=distance_km,
            status="Manual Review",
            confidence="Low",
            remarks="Reverse geocoder returned insufficient data.",
        )

    state_match = bool(input_state) and bool(geocoded_state) and (
        normalize_text(input_state) == normalize_text(geocoded_state)
        or fuzz.token_sort_ratio(normalize_text(input_state), normalize_text(geocoded_state)) >= 90
    )
    district_score = district_similarity(input_district, geocoded_district)
    district_match = district_score >= settings.district_fuzzy_threshold
    input_pin_clean = clean_pincode(input_pincode)
    geocoded_pin_clean = clean_pincode(geocoded_pincode)
    pin_match = bool(input_pin_clean) and input_pin_clean == geocoded_pin_clean

    remarks_parts = []
    if not state_match:
        remarks_parts.append("state mismatch")
    if not district_match:
        remarks_parts.append(f"district mismatch (similarity {district_score:.0f}%)")
    if not pin_match:
        remarks_parts.append(
            "PIN mismatch" if input_pin_clean and geocoded_pin_clean else "PIN not comparable (missing on one side)"
        )
    if distance_km is None:
        remarks_parts.append("could not compute distance from PIN's own reference point")

    if (
        distance_km is not None
        and pin_match and district_match and state_match
        and distance_km < settings.accurate_distance_km
    ):
        status = "Accurate"
    elif (
        district_match and distance_km is not None
        and settings.accurate_distance_km <= distance_km < settings.nearby_distance_km
    ):
        status = "Nearby"
    else:
        status = "Incorrect"
        if district_match and pin_match and state_match and distance_km is not None:
            remarks_parts.append(f"distance {distance_km:.1f}km exceeds nearby threshold despite matching fields")

    score = (
        (2 if pin_match else 0)
        + (2 if district_match else 0)
        + (1 if state_match else 0)
        + (
            2 if distance_km is not None and distance_km < settings.accurate_distance_km
            else 1 if distance_km is not None and distance_km < settings.nearby_distance_km
            else 0
        )
    )
    confidence = "High" if score >= 6 else "Medium" if score >= 3 else "Low"

    return ComparisonResult(
        state_match=state_match,
        district_match=district_match,
        pin_match=pin_match,
        distance_km=distance_km,
        status=status,
        confidence=confidence,
        remarks="; ".join(remarks_parts) if remarks_parts else "All checks passed.",
    )
