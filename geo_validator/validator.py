"""Entry point: validates every row of an input Excel of market-yard/pincode
geo-coordinates against reverse geocoding, and writes an annotated output
workbook plus a terminal summary.

CLI:
    python -m geo_validator.validator --input path/to/file.xlsx
    python -m geo_validator.validator --input path/to/file.xlsx --limit 20   # quick smoke test
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from geo_validator.compare import compare_row
from geo_validator.config import SETTINGS, Settings
from geo_validator.geocoder import GeoCache, ReverseGeocoder
from geo_validator.utils import clean_pincode, haversine_km, is_valid_coordinate, setup_logging

# Recognized header spellings per canonical field -- real input files use
# either the "State/District/Market Yard/Pincode" shape or the
# "PIN Code/Post Office/Taluk/District/State" shape (or a hybrid, once
# lat/long have been added), so columns are matched by normalized header
# text rather than assuming one fixed schema.
_COLUMN_SYNONYMS = {
    "state": {"state"},
    "district": {"district"},
    "pincode": {"pincode", "pincodes", "pin", "pin code", "postal_code", "postcode"},
    "latitude": {"latitude", "lat"},
    "longitude": {"longitude", "lon", "lng", "long"},
}
_REQUIRED_COLUMNS = {"state", "district", "pincode", "latitude", "longitude"}


def _detect_columns(columns: list) -> dict:
    normalized = {c: str(c).strip().lower() for c in columns}
    detected = {}
    for canonical, synonyms in _COLUMN_SYNONYMS.items():
        for col, norm in normalized.items():
            if norm in synonyms:
                detected[canonical] = col
                break
    missing = _REQUIRED_COLUMNS - detected.keys()
    if missing:
        raise ValueError(f"Could not detect required column(s) {sorted(missing)} in input file. Found columns: {list(columns)}")
    return detected


def _maps_link(lat, lon) -> str:
    if not is_valid_coordinate(lat, lon):
        return ""
    return f"https://www.google.com/maps?q={lat},{lon}"


def validate_file(
    input_path: Path,
    output_path: Path,
    settings: Settings = SETTINGS,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    df = pd.read_excel(input_path)
    if limit:
        df = df.head(limit)
    cols = _detect_columns(list(df.columns))
    logger.info(f"Detected columns: {cols}")

    cache = GeoCache(settings.cache_path)
    geocoder = ReverseGeocoder(settings, cache)

    out_rows = []
    counts = {"Accurate": 0, "Nearby": 0, "Incorrect": 0, "Manual Review": 0}
    total = len(df)
    t0 = time.monotonic()

    for i, row in df.iterrows():
        lat, lon = row.get(cols["latitude"]), row.get(cols["longitude"])
        input_state = row.get(cols["state"], "")
        input_district = row.get(cols["district"], "")
        input_pincode = row.get(cols["pincode"], "")

        if not is_valid_coordinate(lat, lon):
            logger.warning(f"Row {i}: missing/invalid coordinate ({lat}, {lon})")
            geo = None
        else:
            geo = geocoder.reverse(float(lat), float(lon))

        has_data = geo is not None and geo.has_data
        distance_km = None
        # pandas represents a missing pincode as NaN, which is truthy in
        # Python -- `if input_pincode:` would wrongly proceed and try to
        # geocode the literal string "nan" for every row lacking a real
        # pincode. clean_pincode() strips that down to "" so it's correctly
        # treated as missing (confirmed live: only 165/5737 last_known_prices
        # rows actually carry a pincode -- the rest are nationwide entries
        # with lat/long only).
        pincode_clean = clean_pincode(input_pincode)
        if has_data and pincode_clean:
            ref_point = geocoder.geocode_pincode(pincode_clean, str(input_state))
            if ref_point:
                distance_km = haversine_km(float(lat), float(lon), ref_point[0], ref_point[1])
            else:
                logger.warning(f"Row {i}: could not forward-geocode PIN {pincode_clean!r} for distance check")

        result = compare_row(
            input_state=str(input_state),
            input_district=str(input_district),
            input_pincode=str(input_pincode),
            geocoded_state=geo.state if geo else "",
            geocoded_district=geo.district if geo else "",
            geocoded_pincode=geo.postcode if geo else "",
            has_geocode_data=has_data,
            distance_km=distance_km,
            settings=settings,
        )
        counts[result.status] += 1

        out_rows.append({
            **row.to_dict(),
            "Verified_State": geo.state if geo else "",
            "Verified_District": geo.district if geo else "",
            "Verified_Taluk": geo.taluk if geo else "",
            "Verified_Village": geo.village if geo else "",
            "Verified_Town": geo.town if geo else "",
            "Verified_City": geo.city if geo else "",
            "Verified_PIN": geo.postcode if geo else "",
            "Formatted_Address": geo.formatted_address if geo else "",
            "Distance_km": round(distance_km, 3) if distance_km is not None else "",
            "State_Match": result.state_match,
            "District_Match": result.district_match,
            "PIN_Match": result.pin_match,
            "Coordinate_Status": result.status,
            "Confidence": result.confidence,
            "Remarks": result.remarks,
            "Google_Maps_Link": _maps_link(lat, lon),
        })

        if (i + 1) % 50 == 0 or (i + 1) == total:
            cache.save()
            elapsed = time.monotonic() - t0
            logger.info(f"Processed {i + 1}/{total} rows ({elapsed:.0f}s elapsed)")

    cache.save()
    out_df = pd.DataFrame(out_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_excel(output_path, index=False)

    accurate_pct = 100 * counts["Accurate"] / total if total else 0.0
    logger.info(
        "\n=== Validation summary ===\n"
        f"Total Records:  {total}\n"
        f"Accurate:       {counts['Accurate']}\n"
        f"Nearby:         {counts['Nearby']}\n"
        f"Incorrect:      {counts['Incorrect']}\n"
        f"Manual Review:  {counts['Manual Review']}\n"
        f"Accuracy %:     {accurate_pct:.1f}%\n"
        f"Output written: {output_path}"
    )
    return out_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate market-yard geo-coordinates against reverse geocoding.")
    parser.add_argument("--input", required=True, type=Path, help="Path to the input Excel file.")
    parser.add_argument("--output", type=Path, default=SETTINGS.default_output_path, help="Path for the annotated output workbook.")
    parser.add_argument("--log", type=Path, default=SETTINGS.default_log_path, help="Path for the failure/error log file.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N rows (for a quick test run).")
    args = parser.parse_args()

    setup_logging(args.log)
    if not SETTINGS.use_google:
        logger.info("GOOGLE_MAPS_API_KEY not set -- falling back to Nominatim (OpenStreetMap), rate-limited to ~1 req/sec.")

    try:
        validate_file(args.input, args.output, SETTINGS, limit=args.limit)
    except Exception:
        logger.exception("Validation run failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
