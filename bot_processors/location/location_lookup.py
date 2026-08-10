"""Resolves a caller's spoken location + pincode into coordinates, then ranks
individual Agmarknet market yards (mandis) by proximity so the bot can read
out nearby candidates for the caller to confirm by name — see
system_prompt.txt's "Caller location" section. Design: ask the caller's
spoken location (village/mandal/district) first, pincode second, then
confirm from a geography-ranked candidate list of actual market yard names
(however many are genuinely nearby, not a fixed count) rather than the bot
guessing a single district and asking yes/no. Market yards, not districts,
are what a farmer actually recognizes by name and what pins down their exact
location most precisely.

Two-tier coordinate lookup, in order:
  1. OpenWeatherMap's ZIP endpoint resolves the caller's pincode directly to
     coordinates, nationwide — used in preference to the AP-only
     postoffice-based pincode_geo_reference.json, which only ever covers
     Andhra Pradesh.
  2. If the pincode doesn't resolve (typo, or OpenWeatherMap has no ZIP data
     for it), falls back to geocoding the caller's spoken place name instead,
     reusing weather_lookup._geocode (the same OpenWeatherMap Direct
     Geocoding call get_weather already relies on).

Nearby-market ranking reuses the same per-market lat/lon rows from
nationwide_market_geo_reference.json / market_yard_geo_reference.json that
enrich_prices_with_geo.py already uses to enrich last_known_prices.json —
market_yard's entries (AP/TG, precise pincode-based coordinates) take
precedence over nationwide's coarser ones for any market present in both.

CLI: none — these are real LLM tool calls, registered on the LLMContext's
`tools=[...]` list in Bot.py (see make_confirm_location/make_save_caller_location).
"""

import math
import os
import re
import time

import json
import requests
from loguru import logger
from rapidfuzz import fuzz, process

from pipecat.services.llm_service import FunctionCallParams

from bot_processors.calls.call_db import log_tool_call
from bot_processors.calls.caller_db import save_location, update_contact_name
from bot_processors.core.task_tracker import track_task
from bot_processors.location.weather_lookup import _geocode as _geocode_place
from bot_processors.paths import DATA_DIR
from bot_processors.pricing.price_lookup import _resolve_state

_OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY", "")

_MARKET_GEO_PATH = DATA_DIR / "market_yard_geo_reference.json"
_NATIONWIDE_GEO_PATH = DATA_DIR / "nationwide_market_geo_reference.json"
_PINCODE_GEO_PATH = DATA_DIR / "pincode_geo_reference.json"

# "Surrounding/neighboring" market yards, not a fixed count: everything
# within the radius, capped so the bot doesn't read out an unreasonably long
# list; if fewer than the minimum fall inside the radius (sparse geocoding in
# that area), the nearest few are offered anyway rather than returning
# nothing.
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


# (state, district, market) -> (lat, lon). Keyed at market granularity, not
# district — a farmer recognizes a specific mandi by name far more reliably
# than an administrative district name, and it pins down their location more
# precisely for storage.
_market_points: dict[tuple[str, str, str], tuple[float, float]] = {}
_market_points_mtimes: tuple[float, float] = (0.0, 0.0)


def _load_market_points() -> dict[tuple[str, str, str], tuple[float, float]]:
    points: dict[tuple[str, str, str], tuple[float, float]] = {}
    # Nationwide first (broad, coarser coverage), then market_yard (AP/TG,
    # precise pincode-based coordinates) so it overwrites any duplicate key
    # with the more precise fix — same precedence enrich_prices_with_geo.py
    # uses between these two files.
    for path in (_NATIONWIDE_GEO_PATH, _MARKET_GEO_PATH):
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)
        for r in rows:
            if r.get("geocode_status") != "ok" or r.get("latitude") is None:
                continue
            key = (r["state"], r["district"], r["market"])
            points[key] = (r["latitude"], r["longitude"])
    return points


def _reload_market_points_if_changed() -> None:
    """Same mtime-guarded reload pattern as price_lookup._reload_last_known_if_changed
    — these reference files are rebuilt by the offline geocoding scripts, not
    by this long-running bot process."""
    global _market_points, _market_points_mtimes
    m1 = _NATIONWIDE_GEO_PATH.stat().st_mtime if _NATIONWIDE_GEO_PATH.exists() else 0.0
    m2 = _MARKET_GEO_PATH.stat().st_mtime if _MARKET_GEO_PATH.exists() else 0.0
    if (m1, m2) == _market_points_mtimes:
        return
    _market_points = _load_market_points()
    _market_points_mtimes = (m1, m2)


def _nearby_markets(state: str, lat: float, lon: float) -> list[tuple[str, str, float]]:
    """Market yards (mandis) of `state` sorted by distance from (lat, lon),
    nearest first, as (market, district, distance_km) — everything within
    _NEARBY_RADIUS_KM, or the closest few if the radius catches too few
    (sparse geocoding coverage in that area)."""
    _reload_market_points_if_changed()
    same_state = sorted(
        (
            (market, district, _haversine_km(lat, lon, m_lat, m_lon))
            for (s, district, market), (m_lat, m_lon) in _market_points.items()
            if s == state
        ),
        key=lambda x: x[2],
    )
    within_radius = [m for m in same_state if m[2] <= _NEARBY_RADIUS_KM]
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


# Confirmed live (call_919154708539_0f7aaf36.log, 2026-08-07): Soniox
# sometimes transcribes spoken pincode digits as Telugu number WORDS
# ("ఐదు ఒకటి ఐదు ఒకటి" = "five one five one") rather than numerals, and the
# caller repeated the whole pincode over a dozen turns because every single
# one of those calls extracted zero digits — isdigit() only recognizes actual
# digit characters, never number words, so the buffer never moved off empty.
# Telugu is the primary/only language this bot speaks (see system_prompt.txt),
# but Soniox's language detection ("lang=auto") can occasionally land on
# English for a turn too, so both tables are checked. Two "zero" spellings
# are included (సున్నా is the canonical/TTS spelling, సున్న is a common
# shorter spoken variant — both were heard live in the same call).
_TELUGU_DIGIT_WORDS = {
    "సున్నా": "0", "సున్న": "0",
    "ఒకటి": "1", "రెండు": "2", "మూడు": "3", "నాలుగు": "4",
    "ఐదు": "5", "ఆరు": "6", "ఏడు": "7", "ఎనిమిది": "8", "తొమ్మిది": "9",
}
_ENGLISH_DIGIT_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}
# Splits on plain whitespace/punctuation only — deliberately NOT a \w-based
# token regex. First attempt used r"[^\W\d_]+|\d+", which silently truncated
# Telugu words: "ఐదు" (ఐ+ద+ు, where ు is a combining vowel sign / matra) came
# back as "ఐద" from re.findall, because Python's \w doesn't treat that
# trailing matra as a word character — so it never matched _TELUGU_DIGIT_WORDS
# at all. Splitting on delimiters instead of matching word-characters sidesteps
# that Unicode-category pitfall completely: each whole word (matras included)
# survives intact as one token.
_TOKEN_SPLIT_RE = re.compile(r"[\s,.\-]+")


def _clean_pincode(pincode: str) -> str:
    """Callers dictate pincodes as fragmented digit groups over several turns
    (e.g. "5, 3, 3" then "4, 0, 1") and STT sometimes inserts stray
    punctuation ("533-401"), or transcribes them as spoken number words
    instead of numerals ("ఐదు ఒకటి" instead of "51") — extract digit
    characters AND recognized digit words (Telugu primary, English as a
    fallback), in the order they appear, so a pincode built up that way
    still resolves. Unrecognized words (filler, names, politeness particles
    like "అండి") are silently skipped, same as stray punctuation always was."""
    out = []
    for token in _TOKEN_SPLIT_RE.split(pincode or ""):
        if not token:
            continue
        if token.isdigit():
            out.append(token)
        elif token in _TELUGU_DIGIT_WORDS:
            out.append(_TELUGU_DIGIT_WORDS[token])
        elif token.lower() in _ENGLISH_DIGIT_WORDS:
            out.append(_ENGLISH_DIGIT_WORDS[token.lower()])
    return "".join(out)


def make_collect_pincode_digits(serializer=None):
    """Builds the collect_pincode_digits direct-function tool — a
    deterministic digit accumulator that replaces asking the LLM to
    "silently concatenate digits across turns" itself.

    That instruction (still present in system_prompt.txt's older wording as
    a fallback) turned out unreliable in practice: real calls showed the
    model losing count and re-asking for the whole pincode even after it had
    already received exactly 6 digits (e.g. caller said "53" then "1118" —
    6 digits total — and the model still said "please give 6 digits").
    Counting/concatenating digits from conversation history is exactly the
    kind of precise bookkeeping LLMs are inconsistent at, especially when
    digits arrive in uneven-sized chunks. This tool moves that bookkeeping
    into plain Python instead: the LLM calls it with whatever digit-ish
    thing the caller just said, every time, and the tool itself tracks how
    many digits have accumulated and only reports "complete" once there are
    really 6 — no arithmetic left for the model to get wrong.

    State is a plain closure variable, not a global dict keyed by call_id —
    make_collect_pincode_digits() is called fresh once per call inside
    run_bot() (see Bot.py), so each call naturally gets its own buffer with
    no cross-call leakage and nothing to clean up afterward.
    """
    _buffer = ""

    async def collect_pincode_digits(params: FunctionCallParams, digits: str = "", reset: bool = False) -> None:
        """Call this every time the caller says anything containing pincode
        digits — even a single digit, or just a few — instead of trying to
        track or concatenate the digits yourself. Call it fresh each turn
        with only what the caller said THIS turn (not what you've already
        submitted before); the tool remembers everything submitted so far
        in this call and does the counting for you.

        - If the result says complete=false, digits_needed tells you how
          many more digits are still needed — just ask for the rest (or
          keep listening) and call this again with the next digits, do NOT
          re-ask for the whole pincode from scratch.
        - If the result says complete=true, "pincode" is the full assembled
          6-digit pincode — immediately call lookup_place_by_pincode (or
          confirm_location, whichever flow you're in) with it.
        - If the caller says they misspoke and wants to restart their
          pincode, call this once with reset=true (digits can be empty) to
          clear what's been collected so far, then start over normally.

        Args:
            digits: Exactly what the caller said this turn that relates to
                their pincode (digits, spoken numbers already transcribed,
                or a mix) — strip nothing yourself, just pass it through.
                Leave this empty ONLY when calling purely to reset (see
                reset below) — every other call must include it.
            reset: Set true to discard everything collected so far in this
                call and start counting over from zero. Safe to combine
                with digits in the same call (e.g. the caller corrects
                themselves and immediately gives the real pincode in one
                breath) — reset is applied first, then digits is added to
                the now-empty buffer.
        """
        nonlocal _buffer
        _t0 = time.monotonic()

        new_digits = _clean_pincode(digits)
        # A single call already carrying 6+ digits is treated as a fresh
        # full pincode, replacing the buffer rather than appending to it —
        # confirmed live: despite the docstring saying "only what the caller
        # said THIS turn," the model sometimes bundles already-submitted
        # digits back in (buffer had '5151', next call passed '24515124').
        # Blindly appending would splice stale+new digits into a pincode
        # nobody actually said; a fresh 6+ digit chunk is far more likely to
        # be the caller/model restating the whole number than a genuine
        # continuation, so this is the safer assumption.
        if reset or len(new_digits) >= 6:
            _buffer = new_digits
        else:
            _buffer = _buffer + new_digits

        if len(_buffer) >= 6:
            pincode, leftover = _buffer[:6], _buffer[6:]
            _buffer = ""  # reset for any subsequent, independent pincode in this same call
            result = {"complete": True, "pincode": pincode}
            outcome = f"complete -> {pincode}" + (f" ({len(leftover)} extra digit(s) dropped)" if leftover else "")
        else:
            result = {"complete": False, "digits_so_far": _buffer, "digits_needed": 6 - len(_buffer)}
            outcome = f"collecting ({len(_buffer)}/6): {_buffer!r}"

        await params.result_callback(result)
        exec_ms = int((time.monotonic() - _t0) * 1000)

        if serializer is not None and serializer.call_id:
            track_task(log_tool_call(
                serializer.call_id, "collect_pincode_digits", f"digits={digits!r} reset={reset}",
                outcome, True, exec_ms,
            ))

    return collect_pincode_digits


# pincode -> list of {village, mandal, district, state}, built from
# pincode_geo_reference.json (itself built from all_ap_postoffices.xlsx —
# see pincode_geo_reference.py). Deliberately NOT joined into
# last_known_prices.json/enrich_prices_with_geo.py's district-centroid data —
# that was tried before and abandoned (see enrich_prices_with_geo.py's
# docstring: attaching every pincode in a district to every market in it gave
# up to 98 pincodes per district, with no way to tell which one actually
# belonged to which market). Keyed at the exact-pincode level instead, which
# is precise: a single pincode covers only a handful of post offices, not a
# whole district.
_pincode_places: dict[str, list[dict]] = {}
_pincode_places_mtime: float = 0.0

# Some pincodes (dense urban areas especially) cover dozens of post offices —
# confirmed live: 470 of 1,159 indexed pincodes have more than 8 (one has 27).
# Reading that many names aloud on a call is unusable, so past this count the
# tool stops offering a pick-list. mandal/district/state are identical across
# every candidate sharing one pincode (they're geocoded per-taluk, not
# per-post-office — see pincode_geo_reference.py) so those three are still
# usable even when the village itself has to be asked for as free text.
_MAX_CANDIDATES_TO_OFFER = 8


def _load_pincode_places() -> dict[str, list[dict]]:
    if not _PINCODE_GEO_PATH.exists():
        return {}
    with open(_PINCODE_GEO_PATH, encoding="utf-8") as f:
        rows = json.load(f)
    by_pincode: dict[str, list[dict]] = {}
    seen: set[tuple[str, str]] = set()
    for r in rows:
        # Only "matched" rows: their district has actually been cross-checked
        # against Agmarknet's real district vocabulary (see
        # pincode_geo_reference.py) — lower-confidence rows are skipped
        # rather than risking a wrong district for the caller's record.
        if r.get("match_confidence") != "matched":
            continue
        pincode, post_office, taluk, district, state = (
            r.get("pincode"), r.get("post_office"), r.get("taluk"),
            r.get("matched_district"), r.get("state"),
        )
        if not pincode or not post_office or not district:
            continue
        dedupe_key = (pincode, post_office.strip().lower())
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        by_pincode.setdefault(pincode, []).append({
            "village": post_office, "mandal": taluk, "district": district, "state": state,
        })
    return by_pincode


def _reload_pincode_places_if_changed() -> None:
    global _pincode_places, _pincode_places_mtime
    mtime = _PINCODE_GEO_PATH.stat().st_mtime if _PINCODE_GEO_PATH.exists() else 0.0
    if mtime == _pincode_places_mtime:
        return
    _pincode_places = _load_pincode_places()
    _pincode_places_mtime = mtime
    _reload_village_index_if_changed(force=True)


# Reverse index for detecting "this village belongs to a DIFFERENT pincode
# than the one the caller gave" — e.g. confirmed live: a caller gave pincode
# 531118 and named a village that doesn't exist there at all, but a
# near-identical name ("Amalapuram") exists under 531117, a different
# pincode in the same district. Without this check, save_caller_location's
# existing "accept whatever village the caller says" fallback (needed for
# genuinely unlisted hamlets) would silently store a village paired with the
# wrong pincode.
#
# Deliberately built from EVERY row regardless of match_confidence — wider
# than _load_pincode_places()'s "matched"-only filter. This index only ever
# produces a soft warning for the caller to confirm, never something saved
# as fact, so a lower-confidence hit is still useful signal here even though
# it isn't trustworthy enough to offer as a normal candidate.
_village_index: list[dict] = []
_village_index_mtime: float = 0.0

# Above this fuzzy-match score (0-100, rapidfuzz.fuzz.ratio), a village name
# is treated as "almost certainly the same place" even without an exact
# string match — tuned to catch single-character STT slips (e.g. a dropped
# or added leading letter: "Amalapuram" vs "Tamalapuram" scores ~95) while
# staying well clear of unrelated village names, which mostly score under 70.
_CROSS_PINCODE_FUZZY_THRESHOLD = 90


def _load_village_index() -> list[dict]:
    if not _PINCODE_GEO_PATH.exists():
        return []
    with open(_PINCODE_GEO_PATH, encoding="utf-8") as f:
        rows = json.load(f)
    index: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for r in rows:
        pincode, post_office, taluk, district, state = (
            r.get("pincode"), r.get("post_office"), r.get("taluk"),
            r.get("matched_district"), r.get("state"),
        )
        if not pincode or not post_office or not district:
            continue
        dedupe_key = (pincode, post_office.strip().lower())
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        index.append({
            "village": post_office, "village_lower": post_office.strip().lower(),
            "pincode": pincode, "mandal": taluk, "district": district, "state": state,
            "confidence": r.get("match_confidence"),
        })
    return index


def _reload_village_index_if_changed(force: bool = False) -> None:
    global _village_index, _village_index_mtime
    mtime = _PINCODE_GEO_PATH.stat().st_mtime if _PINCODE_GEO_PATH.exists() else 0.0
    if not force and mtime == _village_index_mtime:
        return
    _village_index = _load_village_index()
    _village_index_mtime = mtime


def _check_cross_pincode_mismatch(village: str, given_pincode: str) -> dict | None:
    """Returns None if `village` isn't suspiciously tied to some OTHER
    pincode (safe to accept as-is, per save_caller_location's existing
    lenient fallback for unlisted hamlets) — otherwise a dict describing the
    mismatch for the caller to confirm.

    First checks whether the village matches (exact OR fuzzy) something
    already under the GIVEN pincode — if so, always None, no matter how many
    OTHER pincodes also happen to share that village name. This matters:
    confirmed live, common village names like "Annavaram" and "Ramavaram"
    each repeat across 4-10+ AP pincodes, so merely appearing elsewhere is
    not evidence of anything wrong when it's also genuinely present in the
    caller's own stated pincode.

    Only once the village draws a blank in its own pincode do the two
    cross-pincode passes run: exact (case-insensitive) match, for the
    Ramavaram-style case (a real, correctly-spelled village that just
    belongs to a different pincode than the one given); then fuzzy match,
    for the Amalapuram/Tamalapuram-style case (an STT/spelling slip on an
    otherwise-real village name).
    """
    village_norm = (village or "").strip().lower()
    if not village_norm:
        return None
    _reload_village_index_if_changed()

    own = [e for e in _village_index if e["pincode"] == given_pincode]
    if any(e["village_lower"] == village_norm for e in own):
        return None
    if own:
        own_choices = {i: e["village_lower"] for i, e in enumerate(own)}
        own_match = process.extractOne(village_norm, own_choices, scorer=fuzz.ratio,
                                        score_cutoff=_CROSS_PINCODE_FUZZY_THRESHOLD)
        if own_match is not None:
            return None

    others = [e for e in _village_index if e["pincode"] != given_pincode]
    if not others:
        return None

    exact_hits = [e for e in others if e["village_lower"] == village_norm]
    if exact_hits:
        distinct_pincodes = {e["pincode"] for e in exact_hits}
        # Real village names repeat across many pincodes/districts in India
        # (this exact case: "Ramavaram" is a real village under 4 different
        # AP pincodes) — naming ONE as "the" correct pincode when several
        # match would be false precision, so only surface a specific
        # suggested_pincode when there's a single unambiguous hit.
        sample = exact_hits[0]
        return {
            "match_type": "exact",
            "given_village": village, "given_pincode": given_pincode,
            "suggested_pincode": sample["pincode"] if len(distinct_pincodes) == 1 else None,
            "possible_pincodes": sorted(distinct_pincodes),
            "mandal": sample["mandal"], "district": sample["district"], "state": sample["state"],
        }

    choices = {i: e["village_lower"] for i, e in enumerate(others)}
    match = process.extractOne(village_norm, choices, scorer=fuzz.ratio,
                                score_cutoff=_CROSS_PINCODE_FUZZY_THRESHOLD)
    if match is None:
        return None
    _, score, idx = match
    entry = others[idx]
    return {
        "match_type": "fuzzy", "score": round(score, 1),
        "given_village": village, "given_pincode": given_pincode,
        "suggested_pincode": entry["pincode"], "suggested_village": entry["village"],
        "possible_pincodes": [entry["pincode"]],
        "mandal": entry["mandal"], "district": entry["district"], "state": entry["state"],
    }


def _find_own_pincode_match(village: str, given_pincode: str) -> dict | None:
    """If `village` is a real, exact match for one of the GIVEN pincode's
    own rows — including the lower-confidence "matched_geo_outlier" rows
    that lookup_place_by_pincode deliberately never reads aloud as
    candidates (see _load_pincode_places) — returns that row's own
    mandal/district/state. None otherwise.

    This is what makes it safe for lookup_place_by_pincode to only speak the
    high-confidence subset of a pincode's villages (confirmed live: pincode
    515101 has 9 real post offices but only 4 pass the confidence bar) —
    a caller who already knows and states one of the other 5 by name still
    gets it saved correctly, with THAT row's own district rather than
    whatever district the spoken candidates happened to share. Without this,
    save_caller_location would otherwise just save whichever district the
    caller-facing conversation already had in hand, which can be wrong for
    exactly the geographic-outlier rows this filter exists to protect
    against.
    """
    village_norm = (village or "").strip().lower()
    if not village_norm:
        return None
    _reload_village_index_if_changed()
    for e in _village_index:
        if e["pincode"] == given_pincode and e["village_lower"] == village_norm:
            return {"village": e["village"], "mandal": e["mandal"],
                    "district": e["district"], "state": e["state"],
                    "confidence": e["confidence"]}
    return None


def _known_villages_for_pincode(given_pincode: str) -> list[str]:
    """Every village name on record for this pincode, any confidence tier —
    used only for save_caller_location's village_not_recognized prompt (offer
    the caller a "pick the nearest" list), never for lookup_place_by_pincode's
    own spoken candidates (which stay confidence-filtered)."""
    _reload_village_index_if_changed()
    return [e["village"] for e in _village_index if e["pincode"] == given_pincode]


def make_lookup_place_by_pincode(serializer=None):
    """Builds the lookup_place_by_pincode direct-function tool — the
    metadata-capture flow's pincode -> village/mandal/district/state step
    (AP only, for now; see system_prompt.txt's metadata-capture section)."""

    async def lookup_place_by_pincode(params: FunctionCallParams, pincode: str) -> None:
        """Look up the villages known for the caller's pincode, so they can
        confirm or pick which one is theirs.

        Call this as soon as the caller has given a 6-digit pincode, right
        after they've given their name.
        - If it returns exactly one candidate, read it back for a yes/no
          confirmation.
        - If it returns several (a "candidates" list), read out the village
          names and ask which one is theirs — same pattern as
          confirm_location's market-yard list.
        - If it returns "too_many_to_list" instead of a candidates list, do
          NOT read anything out by default — mandal/district/state are
          already known (given in the result), just ask the caller to say
          their village name directly and use whatever they say as-is. The
          result also includes "all_villages", the full list — this is ONLY
          for the caller explicitly doubting/asking to hear every name
          (e.g. "naa vooru ledu ekkada", "andarini cheppu"), not something to
          read proactively. If asked, read all_villages in batches of about
          8 at a time with a quick checkback ("ఇందులో మీది ఉందా? లేదంటే ఇంకా
          చెప్తాను") between batches, rather than one unbroken list.
        - If it returns an error (pincode not covered — currently AP only),
          tell the caller you couldn't find it and ask them to say their
          village, mandal, district, and state directly instead.

        Args:
            pincode: The 6-digit postal pincode the caller said, digits only
                (strip any spaces or punctuation).
        """
        _t0 = time.monotonic()
        _reload_pincode_places_if_changed()

        clean_pincode = _clean_pincode(pincode)
        candidates = _pincode_places.get(clean_pincode, []) if len(clean_pincode) == 6 else []
        exec_ms = int((time.monotonic() - _t0) * 1000)

        if not candidates:
            await params.result_callback({
                "error": "No village data found for this pincode — ask the caller to say their "
                         "village, mandal, district, and state directly instead.",
            })
            outcome = "no match"
        elif len(candidates) > _MAX_CANDIDATES_TO_OFFER:
            # Too many post offices to read aloud by default — mandal/
            # district/state are still known (identical across all of them),
            # so the village itself is normally just taken from the caller as
            # free text. all_villages is included anyway (not spoken unless
            # asked) so the bot can answer on request without a second lookup
            # round-trip — see docstring above.
            sample = candidates[0]
            await params.result_callback({
                "pincode": clean_pincode,
                "too_many_to_list": True,
                "candidate_count": len(candidates),
                "mandal": sample["mandal"], "district": sample["district"], "state": sample["state"],
                "all_villages": [c["village"] for c in candidates],
            })
            outcome = f"too many ({len(candidates)}) — mandal/district/state only"
        else:
            await params.result_callback({"pincode": clean_pincode, "candidates": candidates})
            outcome = ", ".join(c["village"] for c in candidates)

        if serializer is not None and serializer.call_id:
            track_task(log_tool_call(
                serializer.call_id, "lookup_place_by_pincode", f"pincode={pincode!r}", outcome,
                bool(candidates), exec_ms,
            ))

    return lookup_place_by_pincode


def make_confirm_location(serializer=None):
    """Builds the confirm_location direct-function tool — same
    serializer-closure pattern as make_get_price/make_get_weather, so tool
    outcomes are logged with the right call_id."""

    async def confirm_location(params: FunctionCallParams, pincode: str, place: str, state: str) -> None:
        """Resolve the caller's location into a short list of nearby
        Agmarknet market yards (mandis) for them to confirm.

        Call this as soon as you have BOTH the caller's spoken location
        (village, town, or mandal/district name) AND something that looks
        like a 6-digit pincode — even if you're not fully certain you heard
        every digit right, call this with your best-guess digits rather than
        asking the caller to repeat the whole pincode again; if it's wrong
        this will simply fail to resolve and you can ask them to repeat it.
        Never call it with just one of the two, and never call it if the
        caller's location is already known from a system note. This does not
        save anything yet; it only returns candidate market yard names for
        you to read aloud and ask the caller which one is theirs — a farmer
        recognizes a specific mandi by name far more reliably than a
        district name, so always offer market yards, not districts. Once
        they confirm one by name, call save_caller_location with that exact
        market and its district (both given in this result).

        Args:
            pincode: The 6-digit postal pincode the caller said, digits only
                (strip any spaces or punctuation).
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

        clean_pincode = _clean_pincode(pincode)
        coords = _geocode_pincode(clean_pincode) if len(clean_pincode) == 6 else None
        resolved_from = "pincode"
        if coords is None:
            coords = _geocode_place(f"{place}, {resolved_state}")
            resolved_from = "place name"
        if coords is None:
            await params.result_callback({
                "error": "Could not resolve this location — ask the caller to repeat their village/town and pincode.",
            })
            return

        candidates = _nearby_markets(resolved_state, *coords)
        exec_ms = int((time.monotonic() - _t0) * 1000)

        if not candidates:
            await params.result_callback({
                "error": f"No known Agmarknet market yards found near this location in {resolved_state}.",
            })
        else:
            await params.result_callback({
                "state": resolved_state,
                "resolved_from": resolved_from,
                "candidate_markets": [
                    {"market": m, "district": d} for m, d, _ in candidates
                ],
            })

        if serializer is not None and serializer.call_id:
            outcome = ", ".join(m for m, _, _ in candidates) if candidates else "no match"
            track_task(log_tool_call(
                serializer.call_id, "confirm_location",
                f"pincode={pincode!r} place={place!r} state={state!r}", outcome, bool(candidates), exec_ms,
            ))

    return confirm_location


def make_save_caller_location(serializer=None, context=None, confirmed_location_state=None):
    """Builds the save_caller_location direct-function tool.

    context, if given, gets a "confirmed farming location" system message
    injected the instant a save actually succeeds — the same literal message
    text Bot.py's _greet_and_inject_memory injects for a RETURNING caller
    whose location was already on file. Relying on the LLM to instead notice
    and remember its own earlier save_caller_location tool-call result a few
    turns back turned out unreliable in practice: confirmed live
    (call_919949070894_4d35a21e.log, 2026-08-07) — a caller's location saved
    successfully, the bot acknowledged it, and two turns later, asked for a
    price, it asked "which district market do you want" as if the save had
    never happened, forcing the caller to repeat themselves ("మీకు ఆల్రెడీ నా
    ప్రాంతమే చెప్పాను కదా"). A system-prompt instruction alone
    (system_prompt.txt's "the instant save_caller_location succeeds...")
    didn't fix this reliably.

    confirmed_location_state, if given, is a mutable dict (Bot.py's
    _confirmed_location) that this populates on success too — even the
    one-time context injection above turned out insufficient on its own
    (confirmed live, call_919390427476_03c11c0d.log, 2026-08-07: the LLM
    ignored that exact injected message and asked for district again just
    one turn later). Bot.py's on_user_turn_started reads this dict to
    re-inject a short reminder on EVERY subsequent turn instead of just
    once — see its own comment for why that's the more reliable fix."""

    async def save_caller_location(
        params: FunctionCallParams,
        state: str,
        district: str,
        pincode: str,
        mandal: str = "",
        village: str = "",
        market: str = "",
        confirmed_despite_mismatch: bool = False,
    ) -> None:
        """Save the caller's state/district/mandal/village/pincode so future
        calls from this phone number don't need to ask for their location
        again.

        Call this once you have state, district, and pincode at minimum —
        include mandal/village too whenever the caller has given them (the
        normal metadata-capture sequence collects all five). market is
        separate and optional: only pass it when the caller has verbally
        confirmed a specific Agmarknet market yard (from confirm_location's
        candidate list) as part of a price lookup — omit it otherwise.

        Before saving, this checks whether `village` actually belongs to a
        DIFFERENT pincode than `pincode` in our records (a real village name,
        just under the wrong pincode — e.g. the caller misspoke or misheard
        one of the two). If so it does NOT save and instead returns
        village_pincode_mismatch=true:
        - Tell the caller their village name doesn't match the pincode they
          gave, and ask them to double-check one or the other — do NOT guess
          which one is right yourself.
        - If the result includes a single suggested_pincode, you may mention
          it as a possibility ("ఇది <pincode> పిన్ కోడ్‌లో ఉన్నట్టుగా ఉంది,
          అదేనా?"), but if possible_pincodes has more than one entry, the
          village name is genuinely ambiguous (real village names repeat
          across many pincodes) — don't name any single one, just ask the
          caller to reconfirm both pincode and village.
        - If the caller insists their answer is correct anyway (a real,
          unlisted place can coincidentally share a name with an indexed
          village elsewhere), call save_caller_location again with the exact
          same values plus confirmed_despite_mismatch=true to save it as
          given, skipping this check.

        Separately, if the village doesn't match this pincode OR any other
        pincode at all (a name we have no record of anywhere), this returns
        village_not_recognized=true instead of saving, with known_villages —
        every real village on record for the pincode they gave:
        - Read known_villages out to the caller (in batches of about 8 with
          a check-back between batches, same style as lookup_place_by_pincode's
          all_villages) and ask if their village is among these, or which one
          is nearest to theirs.
        - If they pick one, call save_caller_location again with that exact
          village name.
        - If none of them are close and the caller insists on their original
          answer, call save_caller_location again with the same values plus
          confirmed_despite_mismatch=true to save it as given.

        Args:
            state: The state/UT the caller is calling from.
            district: The caller's district.
            pincode: The 6-digit pincode the caller gave.
            mandal: The mandal/taluk the caller named, if given.
            village: The village or town the caller named, if given.
            market: The exact market yard name the caller confirmed via
                confirm_location, if this call is part of that flow.
            confirmed_despite_mismatch: Set true only on a retry, after the
                caller has explicitly confirmed their village/pincode despite
                a village_pincode_mismatch or village_not_recognized result on
                the previous attempt.
        """
        _t0 = time.monotonic()
        phone_number = serializer.caller_number if serializer else ""
        resolved_state = _resolve_state(state) or state
        clean_pincode = _clean_pincode(pincode)
        corrected_from_own_pincode = False

        if village and pincode and not confirmed_despite_mismatch:
            mismatch = _check_cross_pincode_mismatch(village, clean_pincode)
            if mismatch is not None:
                await params.result_callback({
                    "village_pincode_mismatch": True,
                    **mismatch,
                })
                if serializer is not None and serializer.call_id:
                    track_task(log_tool_call(
                        serializer.call_id, "save_caller_location",
                        f"{village}, {resolved_state} ({pincode})",
                        f"mismatch flagged ({mismatch['match_type']}, "
                        f"possible={mismatch['possible_pincodes']})",
                        False, int((time.monotonic() - _t0) * 1000),
                    ))
                return

            # Not a cross-pincode mismatch — but if the caller named one of
            # this SAME pincode's own villages that lookup_place_by_pincode
            # didn't read aloud (a matched_geo_outlier row), use THAT row's
            # own district/mandal rather than whatever the conversation
            # already had in hand from the spoken candidates, which can be
            # wrong for exactly these rows. See _find_own_pincode_match.
            own_match = _find_own_pincode_match(village, clean_pincode)
            if own_match is not None:
                if own_match["district"] != district or own_match["mandal"] != mandal:
                    logger.info(
                        f"📍 save_caller_location: correcting district/mandal for "
                        f"{village!r} (pincode {clean_pincode}) from "
                        f"{district!r}/{mandal!r} to {own_match['district']!r}/"
                        f"{own_match['mandal']!r} — matched a {own_match['confidence']} "
                        f"row not offered as a spoken candidate"
                    )
                    district, mandal = own_match["district"], own_match["mandal"]
                    corrected_from_own_pincode = True
            else:
                # Not found under this pincode, not found under any other
                # pincode either — genuinely no record of this name anywhere.
                # Rather than silently trusting an unverifiable spelling, let
                # the caller pick the nearest real village from what we
                # actually have on file for their pincode.
                known = _known_villages_for_pincode(clean_pincode)
                if known:
                    await params.result_callback({
                        "village_not_recognized": True,
                        "given_village": village, "given_pincode": clean_pincode,
                        "known_villages": known,
                        "mandal": mandal, "district": district, "state": resolved_state,
                    })
                    if serializer is not None and serializer.call_id:
                        track_task(log_tool_call(
                            serializer.call_id, "save_caller_location",
                            f"{village}, {resolved_state} ({pincode})",
                            f"not recognized — offered {len(known)} known villages",
                            False, int((time.monotonic() - _t0) * 1000),
                        ))
                    return

        ok = await save_location(phone_number, resolved_state, district, mandal, village, pincode, market)
        exec_ms = int((time.monotonic() - _t0) * 1000)

        if ok:
            await params.result_callback({
                "saved": True, "state": resolved_state, "district": district,
                "mandal": mandal, "village": village, "pincode": pincode,
            })
            if context is not None:
                village_note = f", village {village}" if village else ""
                mandal_note = f", mandal {mandal}" if mandal else ""
                pincode_note = f" (pincode {pincode})" if pincode else ""
                market_note = f", nearest market yard {market}" if market else ""
                context.add_message({
                    "role": "system",
                    "content": (
                        f"This caller's confirmed farming location is {district}, "
                        f"{resolved_state}{mandal_note}{village_note}{pincode_note}{market_note}. "
                        "This caller's name and location metadata are already known — do not ask for "
                        "their name, state, district, mandal, village, or pincode again this call. "
                        "Use this district/state automatically for get_price and get_weather whenever "
                        "the caller doesn't name a different place this turn."
                    ),
                })
            if confirmed_location_state is not None:
                confirmed_location_state.update({
                    "district": district, "state": resolved_state, "village": village,
                })
        else:
            await params.result_callback({"error": "Could not save the location right now."})

        if serializer is not None and serializer.call_id:
            outcome = "saved" if ok else "failed"
            if corrected_from_own_pincode:
                outcome += " (district/mandal auto-corrected to matched_geo_outlier row)"
            track_task(log_tool_call(
                serializer.call_id, "save_caller_location",
                f"{village or mandal or district}, {resolved_state} ({pincode})",
                outcome, ok, exec_ms,
            ))

    return save_caller_location


def make_save_caller_name(serializer=None):
    """Builds the save_caller_name direct-function tool — captures the
    caller's name immediately as part of the metadata-capture sequence,
    rather than waiting for it to be inferred later from the post-call
    summary (see caller_summarizer.py/update_contact_name's other caller)."""

    async def save_caller_name(params: FunctionCallParams, name: str) -> None:
        """Save the caller's name as soon as they've given it.

        Call this once, right after the caller states their name, before
        moving on to asking their state/district/mandal/village/pincode.

        Args:
            name: The caller's name, exactly as they said it.
        """
        _t0 = time.monotonic()
        phone_number = serializer.caller_number if serializer else ""

        await update_contact_name(phone_number, name)
        exec_ms = int((time.monotonic() - _t0) * 1000)

        await params.result_callback({"saved": True, "name": name})

        if serializer is not None and serializer.call_id:
            track_task(log_tool_call(
                serializer.call_id, "save_caller_name", name, "saved", True, exec_ms,
            ))

    return save_caller_name
