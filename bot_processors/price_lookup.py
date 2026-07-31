"""Mandi (market) price lookup: a real LLM tool call. The LLM extracts
whatever commodity, state, and district/town the caller said (or asks for
whichever is missing) and calls get_price(commodity=..., state=..., district=...)
directly — no commodity or location is ever hardcoded here; every commodity
and every state/UT this bot can answer about comes from the live site itself.

Prices come primarily from the persisted "last known price" cache
(bot_processors/last_known_prices.json), written by
bot_processors.agmarknet_scraper's scheduled background job, which
dynamically covers every state/UT and every commodity Agmarknet tracks —
nothing hand-picked. On a genuine cache miss for a given (state, commodity,
district) — most likely just a race with the background job's own refresh
cycle — fetch_price_for_district() below falls back to a live, one-time
scrape (bot_processors.agmarknet_scraper.scrape_single) — see that
function's docstring for the full mechanism.

Unlike the previous keyword/alias-dict version of this module, commodity,
state, and district are never matched against raw sentence text here —
the LLM already extracts each as a clean, isolated tool argument, so
resolution is a straightforward fuzzy match of a short term against a
reference list (see _fuzzy_match), not a word-boundary scan through a
whole transcript.
"""

import asyncio
import difflib
import json
import time
from datetime import date

from loguru import logger

from pipecat.services.llm_service import FunctionCallParams

from bot_processors.agmarknet_scraper import (
    _COMMODITY_REFERENCE_PATH,
    _STATE_REFERENCE_PATH,
    load_commodity_reference,
    load_state_reference,
    save_scraped,
    scrape_single,
)
from bot_processors.call_db import log_tool_call
from bot_processors.price_shared import (
    _LAST_KNOWN_KEY_SEP,
    _LAST_KNOWN_PATH,
    crop_keyword_for,
    normalize_commodity_name,
)
from bot_processors.task_tracker import track_task


def _fuzzy_match(term: str, candidates: list[str]) -> str | None:
    """Matches a short, already-isolated term (an LLM tool-call argument,
    not a whole sentence) against a reference list: exact (case-insensitive)
    first, then prefix (handles a caller/LLM giving a shorter or longer form
    than the catalog, e.g. "cashew" vs. catalog's "Cashewnuts"), then
    difflib for spelling/STT noise. Returns the candidate in its original
    casing, or None if nothing is close enough."""
    lowered = term.strip().lower()
    if not lowered or not candidates:
        return None

    by_lower = {c.lower(): c for c in candidates}
    if lowered in by_lower:
        return by_lower[lowered]

    prefix_matches = [c for c in candidates if c.lower().startswith(lowered) or lowered.startswith(c.lower())]
    if prefix_matches:
        return min(prefix_matches, key=len)

    close = difflib.get_close_matches(lowered, list(by_lower.keys()), n=1, cutoff=0.6)
    if close:
        return by_lower[close[0]]

    return None


# ── Commodity reference (harvested live from Agmarknet's own dropdowns) ────
# bot_processors.agmarknet_scraper.harvest_commodity_reference() walks the
# site's real Commodity Group/Commodity dropdowns — ~600 commodities, not a
# hand-picked list — so any commodity Agmarknet tracks can be recognized.
_commodity_reference: dict[str, tuple[str, str]] = {}
_commodity_reference_mtime: float = 0.0


def _reload_commodity_reference_if_changed() -> None:
    global _commodity_reference, _commodity_reference_mtime
    path = _COMMODITY_REFERENCE_PATH
    if not path.exists():
        return
    mtime = path.stat().st_mtime
    if mtime == _commodity_reference_mtime:
        return
    raw = load_commodity_reference()
    lookup: dict[str, tuple[str, str]] = {}
    for group, commodities in raw.items():
        for commodity in commodities:
            lookup[normalize_commodity_name(commodity)] = (group, commodity)
    _commodity_reference = lookup
    _commodity_reference_mtime = mtime


def _resolve_commodity(term: str) -> tuple[str, str, str] | None:
    """Returns (crop_keyword, group, commodity) for the caller's named
    commodity, matched against the full harvested reference, or None."""
    _reload_commodity_reference_if_changed()
    if not _commodity_reference:
        return None
    match = _fuzzy_match(term, list(_commodity_reference.keys()))
    if match is None:
        return None
    group, commodity = _commodity_reference[match]
    return (crop_keyword_for(commodity), group, commodity)


# ── State reference (harvested live from Agmarknet's State/UT dropdown) ────
_state_reference: list[str] = []
_state_reference_mtime: float = 0.0


def _reload_state_reference_if_changed() -> None:
    global _state_reference, _state_reference_mtime
    path = _STATE_REFERENCE_PATH
    if not path.exists():
        return
    mtime = path.stat().st_mtime
    if mtime == _state_reference_mtime:
        return
    _state_reference = load_state_reference()
    _state_reference_mtime = mtime


def _resolve_state(term: str) -> str | None:
    """Returns the exact state/UT string Agmarknet's dropdown expects for
    whatever state name the caller (or the LLM, inferring one from a
    well-known district) gave, or None if it doesn't match any real
    state/UT."""
    _reload_state_reference_if_changed()
    return _fuzzy_match(term, _state_reference)


def _load_last_known() -> dict[tuple[str, str, str], dict]:
    if not _LAST_KNOWN_PATH.exists():
        return {}
    try:
        raw = json.loads(_LAST_KNOWN_PATH.read_text(encoding="utf-8"))
        return {tuple(k.split(_LAST_KNOWN_KEY_SEP)): v for k, v in raw.items()}
    except Exception:
        logger.opt(exception=True).warning(
            "⚠️ price_lookup: failed to load last-known price cache, starting empty"
        )
        return {}


_last_known: dict[tuple[str, str, str], dict] = _load_last_known()
_last_known_mtime: float = _LAST_KNOWN_PATH.stat().st_mtime if _LAST_KNOWN_PATH.exists() else 0.0


def _reload_last_known_if_changed() -> None:
    """bot_processors.agmarknet_scraper runs as a separate scheduled process
    and rewrites last_known_prices.json independently of this (long-running)
    bot process. Without re-checking the file's mtime on every lookup, a call
    could keep serving whatever was cached at bot startup indefinitely,
    ignoring however many fresh scraper runs happened since — defeating the
    whole point of scraping live data."""
    global _last_known, _last_known_mtime
    if not _LAST_KNOWN_PATH.exists():
        return
    mtime = _LAST_KNOWN_PATH.stat().st_mtime
    if mtime != _last_known_mtime:
        _last_known = _load_last_known()
        _last_known_mtime = mtime


async def fetch_price(crop_keyword: str, location: tuple[str, str]) -> dict | None:
    """Returns the last known price row for this crop/district from the
    persisted cache, or None if nothing has ever been recorded for it.
    "stale" reflects whether that row's own arrival_date is today's date —
    a price the scraper just pulled today is spoken as current, not
    caveated as an old leftover."""
    _reload_last_known_if_changed()
    state, district = location
    last_known_key = (state, district, crop_keyword)
    cached = _last_known.get(last_known_key)
    if not cached:
        return None
    is_stale = cached.get("arrival_date") != date.today().strftime("%d-%m-%Y")
    return {**cached, "stale": is_stale}


def _merge_into_last_known(scraped: dict[str, dict]) -> None:
    """Folds a scrape_single() result straight into the in-memory cache (so
    this same call can use it immediately, without waiting for the next
    _reload_last_known_if_changed() mtime check) and persists it to disk via
    agmarknet_scraper.save_scraped so every district in that state benefits
    going forward, not just whichever one the caller was in."""
    global _last_known, _last_known_mtime
    for key, result in scraped.items():
        _last_known[tuple(key.split(_LAST_KNOWN_KEY_SEP))] = result
    save_scraped(scraped)
    if _LAST_KNOWN_PATH.exists():
        _last_known_mtime = _LAST_KNOWN_PATH.stat().st_mtime


# Live fetches currently in flight, keyed by (state, crop_keyword), so two
# callers asking about the same not-yet-cached commodity in the same state
# at the same time share one scrape_single() browser run instead of each
# launching their own.
_inflight_live_fetches: dict[tuple[str, str], asyncio.Task] = {}


async def _live_fetch_commodity(state: str, group: str, commodity: str, crop_keyword: str) -> dict[str, dict]:
    key = (state, crop_keyword)
    task = _inflight_live_fetches.get(key)
    if task is None:
        task = asyncio.create_task(asyncio.to_thread(scrape_single, state, group, commodity, crop_keyword))
        _inflight_live_fetches[key] = task
        task.add_done_callback(lambda _t, k=key: _inflight_live_fetches.pop(k, None))
    return await task


async def fetch_price_for_district(
    crop_keyword: str, group: str, commodity_name: str, state: str, district_query: str
) -> dict | None:
    """Resolves district_query against whatever districts this (state, crop)
    combo actually has data for, then returns fetch_price() for the matched
    one — or None if nothing matches even after a live fetch.

    District names are never hand-maintained: Agmarknet's form only takes a
    State (district stays "All Districts"), so one query already returns
    every district's rows for that state at once. This first tries a fuzzy
    match against whatever's already cached for this (state, crop); only if
    that comes up empty does it pay for a live scrape_single() call to get
    the current real district list for this state/commodity and try again —
    same "only live-fetch on a genuine miss" principle the old exact-key
    cache lookup used, just re-expressed at fuzzy-match granularity so a
    district that was simply never cached yet (rather than one Agmarknet
    has no data for at all) still gets a fair live attempt instead of
    silently reporting "no data"."""
    _reload_last_known_if_changed()
    cached_candidates = sorted({d for (s, d, c) in _last_known if s == state and c == crop_keyword})
    matched_district = _fuzzy_match(district_query, cached_candidates) if cached_candidates else None

    if matched_district is None:
        logger.info(f"🔴 price_lookup: live fallback fetch for {crop_keyword!r} ({commodity_name}) in {state}")
        scraped = await _live_fetch_commodity(state, group, commodity_name, crop_keyword)
        if scraped:
            _merge_into_last_known(scraped)

        fresh_candidates = sorted({key.split(_LAST_KNOWN_KEY_SEP)[1] for key in scraped})
        matched_district = _fuzzy_match(district_query, fresh_candidates)
        if matched_district is None:
            return None

    return await fetch_price(crop_keyword, (state, matched_district))


def make_get_price(serializer=None):
    """Builds the get_price direct-function tool, closing over this call's
    serializer so tool-call outcomes can still be logged via log_tool_call
    (bot_processors/call_db.py) — same pattern as
    bot_processors.weather_lookup.make_get_weather.
    """

    async def get_price(params: FunctionCallParams, commodity: str, state: str, district: str) -> None:
        """Get the current mandi (market) price for a commodity.

        Call this whenever the caller asks what a crop, vegetable, fruit,
        spice, or other agricultural commodity is currently selling for.
        Only call this once you know the commodity, the state, and the
        district or nearby town — if the caller hasn't given one of these
        yet, do not call this function; ask for whichever is missing first.
        If the caller names a well-known city, town, or district and you
        already know which state it's in, fill in the state yourself
        instead of asking — only ask the caller which state if you are
        genuinely unsure.

        Args:
            commodity: The crop, vegetable, fruit, spice, or other
                commodity the caller asked about, in English or as close to
                their own words as possible (for example "onion", "cotton",
                "cashew"). Never guess a commodity the caller didn't name.
            state: The Indian state or union territory the caller is
                calling from (for example "Andhra Pradesh", "Maharashtra").
            district: The district or nearby town/city the caller named
                (for example "Guntur", "Nashik").
        """
        _t0 = time.monotonic()

        crop_match = _resolve_commodity(commodity)
        if crop_match is None:
            await params.result_callback({
                "error": f"{commodity!r} isn't a commodity Agmarknet tracks — ask the caller to name it differently.",
            })
            return
        crop_keyword, group, commodity_name = crop_match

        resolved_state = _resolve_state(state)
        if resolved_state is None:
            await params.result_callback({
                "error": f"{state!r} isn't a recognized Indian state/UT — ask the caller to confirm which state.",
            })
            return

        result = await fetch_price_for_district(crop_keyword, group, commodity_name, resolved_state, district)
        exec_ms = int((time.monotonic() - _t0) * 1000)

        if result:
            await params.result_callback({
                "commodity": result["commodity"],
                "market": result["market"],
                "district": result["district"],
                "state": resolved_state,
                "arrival_date": result["arrival_date"],
                "modal_per_kg": result["modal_per_kg"],
                "min_per_kg": result["min_per_kg"],
                "max_per_kg": result["max_per_kg"],
                "stale": result["stale"],
            })
        else:
            await params.result_callback({
                "error": (
                    f"No {commodity_name} price data found for {district!r} in {resolved_state} — "
                    "this is a gap in today's government mandi data, not a place that doesn't exist."
                ),
            })

        if serializer is not None and serializer.call_id:
            if result:
                outcome = f"₹{result['modal_per_kg']:.2f}/kg"
                if result["stale"]:
                    outcome += " (stale)"
            else:
                outcome = "no data found"
            track_task(log_tool_call(
                serializer.call_id, "market_price_lookup",
                f"{commodity} in {district}, {state}", outcome, bool(result), exec_ms,
            ))

    return get_price
