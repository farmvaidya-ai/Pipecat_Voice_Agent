"""Mandi (market) price lookup: a real LLM tool call. The LLM extracts
whatever commodity, state, and district/town the caller said (or asks for
whichever is missing) and calls get_price(commodity=..., state=..., district=...)
directly — no commodity or location is ever hardcoded here; every commodity
and every state/UT this bot can answer about comes from the live site itself.

Prices come from the fact_market_prices table (db/schema.sql), written by
bot_processors.agmarknet_scraper's scheduled background job, which
dynamically covers every state/UT and every commodity Agmarknet tracks —
nothing hand-picked. Every lookup here is a live query against that table —
no in-memory cache/mtime-reload dance the way the retired
last_known_prices.json needed, since Postgres is already the shared,
concurrency-safe source of truth. On a genuine miss for a given (state,
commodity, district) — most likely just a race with the background job's
own refresh cycle — fetch_price_for_district() below falls back to a live,
one-time scrape (bot_processors.agmarknet_scraper.scrape_single) — see that
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
import time
from datetime import date

from loguru import logger

from pipecat.adapters.schemas.direct_function import tool_options
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
from bot_processors.db_pool import get_pool
from bot_processors.price_shared import _LAST_KNOWN_KEY_SEP, crop_keyword_for, normalize_commodity_name
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


# How far back a cached price is still trusted enough to speak. Rows are
# real per-day history now (fact_market_prices, db/schema.sql), not a
# single always-overwritten snapshot the way the retired
# last_known_prices.json worked — so with no cutoff at all a caller could
# still be handed the single freshest row on record even if it's weeks old,
# with nothing but a vague "this is a bit old" caveat (confirmed live: a
# real returning-caller summary once cited a price "సుమారు పదిహేను రోజుల
# పాతవి" — about fifteen days old — as if still usable). 4 days matches
# what the caller actually asked for ("last 4 days"): recent-enough-to-be-
# useful data still gets served (with its real date spoken, not a vague
# caveat — see days_old below), older than that is treated the same as no
# data at all rather than silently handed over.
_MAX_STALE_DAYS = 4


def _row_to_price_dict(row) -> dict:
    """Shapes one fact_market_prices row (an asyncpg.Record) into this
    module's response dict — same keys/format every caller here already
    expects (arrival_date as a "%d-%m-%Y" string, matching what the LLM has
    always seen; days_old/stale computed fresh against today, not stored,
    since a row's age changes every day even though the row itself doesn't)."""
    days_old = (date.today() - row["arrival_date"]).days
    result = {
        "commodity": row["commodity"],
        "market": row["market"],
        "district": row["district"],
        "state": row["state"],
        "arrival_date": row["arrival_date"].strftime("%d-%m-%Y"),
        "modal_per_kg": float(row["modal_per_kg"]),
        "min_per_kg": float(row["min_per_kg"]),
        "max_per_kg": float(row["max_per_kg"]),
        "stale": days_old > 0,
        "days_old": days_old,
    }
    if row["arrival_qty"] is not None:
        result["arrival_qty"] = str(row["arrival_qty"])
        result["arrival_unit"] = row["arrival_unit"]
    return result


async def fetch_price(crop_keyword: str, location: tuple[str, str]) -> dict | None:
    """Returns the freshest price row on record for this crop/district
    within _MAX_STALE_DAYS, or None if nothing qualifies (never recorded,
    or everything on record is older than the cutoff — see that constant's
    own comment). "stale" reflects whether the row's arrival_date is today;
    "days_old" is how many days old it actually is (0 when not stale) so
    callers can speak the real age instead of a vague "a bit old"."""
    state, district = location
    pool = get_pool()
    row = await pool.fetchrow(
        """
        SELECT commodity, market, district, state, arrival_date,
               modal_per_kg, min_per_kg, max_per_kg, arrival_qty, arrival_unit
        FROM fact_market_prices
        WHERE state = $1 AND district = $2 AND crop_keyword = $3
          AND arrival_date >= CURRENT_DATE - $4::int
        ORDER BY arrival_date DESC
        LIMIT 1
        """,
        state, district, crop_keyword, _MAX_STALE_DAYS,
    )
    return _row_to_price_dict(row) if row is not None else None


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
    combo has EVER had a price recorded for (any age — this step is purely
    about matching a district NAME, not price freshness, so a district
    whose only price on record is old still gets matched here), then
    returns fetch_price() for the matched one — which applies the real
    _MAX_STALE_DAYS freshness cutoff — or None if nothing matches even
    after a live fetch.

    District names are never hand-maintained: Agmarknet's form only takes a
    State (district stays "All Districts"), so one query already returns
    every district's rows for that state at once. This first tries a fuzzy
    match against whatever's on record for this (state, crop); only if that
    comes up empty, OR the matched district's freshest price turns out to
    be older than _MAX_STALE_DAYS (so fetch_price() itself returns None),
    does it pay for a live scrape_single() call to get the current real
    district list for this state/commodity and try again — same "only
    live-fetch on a genuine miss" principle as before, just against
    fact_market_prices instead of the retired JSON cache."""
    pool = get_pool()
    known_districts = await pool.fetch(
        "SELECT DISTINCT district FROM fact_market_prices WHERE state = $1 AND crop_keyword = $2",
        state, crop_keyword,
    )
    cached_candidates = sorted(r["district"] for r in known_districts)
    matched_district = _fuzzy_match(district_query, cached_candidates) if cached_candidates else None

    if matched_district is not None:
        result = await fetch_price(crop_keyword, (state, matched_district))
        if result is not None:
            return result
        # Matched a district, but its only price on record is too stale to
        # serve (see _MAX_STALE_DAYS) — fall through to a live fetch below
        # instead of giving up here, same as the "never recorded" case.

    logger.info(f"🔴 price_lookup: live fallback fetch for {crop_keyword!r} ({commodity_name}) in {state}")
    scraped = await _live_fetch_commodity(state, group, commodity_name, crop_keyword)
    if scraped:
        await save_scraped(scraped)

    fresh_candidates = sorted({key.split(_LAST_KNOWN_KEY_SEP)[1] for key in scraped})
    matched_district = _fuzzy_match(district_query, fresh_candidates) or matched_district
    if matched_district is None:
        return None

    return await fetch_price(crop_keyword, (state, matched_district))


async def fetch_price_all_markets(crop_keyword: str, state: str) -> list[dict]:
    """Returns the freshest price row (within _MAX_STALE_DAYS) for every
    district this state has a recent price for, for this one commodity —
    the no-district counterpart to fetch_price_for_district(). No live
    scrape is triggered here, since "how many markets have this commodity
    right now" is itself the answer, not a single-district miss to fall
    back from.

    Rows are sorted by modal_per_kg ascending so the caller's own scan (or
    whatever summarizes this list) sees cheapest-first without needing to
    re-sort."""
    pool = get_pool()
    db_rows = await pool.fetch(
        """
        SELECT DISTINCT ON (district) commodity, market, district, state, arrival_date,
               modal_per_kg, min_per_kg, max_per_kg, arrival_qty, arrival_unit
        FROM fact_market_prices
        WHERE state = $1 AND crop_keyword = $2
          AND arrival_date >= CURRENT_DATE - $3::int
        ORDER BY district, arrival_date DESC
        """,
        state, crop_keyword, _MAX_STALE_DAYS,
    )
    rows = [_row_to_price_dict(r) for r in db_rows]
    rows.sort(key=lambda r: r["modal_per_kg"])
    return rows


def make_get_price(serializer=None):
    """Builds the get_price direct-function tool, closing over this call's
    serializer so tool-call outcomes can still be logged via log_tool_call
    (bot_processors/call_db.py) — same pattern as
    bot_processors.weather_lookup.make_get_weather.
    """

    # cancel_on_interruption=False: without this, a caller talking again
    # while this is still fetching (a live scrape can take several seconds)
    # gets the in-flight call cancelled by pipecat, which injects a literal
    # "CANCELLED" tool result into context — NOT the same as "no data", but
    # confirmed live (call_919949070894_4d35a21e.log, 2026-08-07) the LLM
    # treated it that way twice in the same call, telling the caller onion
    # and tomato prices "aren't available right now" when the lookup had
    # simply never finished. A system_prompt.txt instruction telling the LLM
    # how to interpret CANCELLED was tried first and didn't hold up reliably
    # in practice — same LLM-instruction-following gap as the reason
    # collect_pincode_digits and save_caller_location's context injection
    # exist. This is the structural fix instead: the lookup itself just
    # keeps running across the interruption and delivers its real result
    # (via pipecat's async-tool message protocol) once it's actually done,
    # so there's never a fabricated negative answer to give in the first
    # place.
    @tool_options(cancel_on_interruption=False)
    async def get_price(params: FunctionCallParams, commodity: str, state: str, district: str) -> None:
        """Get the current mandi (market) price for a commodity.

        Call this whenever the caller asks what a crop, vegetable, fruit,
        spice, or other agricultural commodity is currently selling for, or
        how much of it arrived at the market (arrival_qty/arrival_unit are
        included in the result when the market reported one). Only call
        this once you know the commodity, the state, and the district or
        nearby town — if the caller hasn't given one of these yet, do not
        call this function; ask for whichever is missing first.
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

        try:
            result = await fetch_price_for_district(crop_keyword, group, commodity_name, resolved_state, district)
        except Exception:
            # fetch_price_for_district's live-scrape fallback (scrape_single,
            # a real Playwright browser launch) can fail for reasons that
            # have nothing to do with whether this commodity/district
            # combo has data — a missing/mismatched browser binary,
            # network hiccup, or Agmarknet layout change. Confirmed live
            # (call_919390427476_391524f9.log, 2026-08-07): this used to
            # propagate straight out of get_price uncaught, which meant
            # result_callback was never called at all — the LLM was left
            # having already told the caller "let me check", with no
            # result and no error ever arriving to react to, so the call
            # just sat silent until the idle-timeout hangup fired. Catching
            # here guarantees the LLM always gets *something* back to
            # apologize with instead of stranding the caller.
            logger.opt(exception=True).error(
                f"🔴 price_lookup: get_price crashed for {commodity_name} in {district}, {resolved_state}"
            )
            result = None
        exec_ms = int((time.monotonic() - _t0) * 1000)

        if result:
            response = {
                "commodity": result["commodity"],
                "market": result["market"],
                "district": result["district"],
                "state": resolved_state,
                "arrival_date": result["arrival_date"],
                "modal_per_kg": result["modal_per_kg"],
                "min_per_kg": result["min_per_kg"],
                "max_per_kg": result["max_per_kg"],
                "stale": result["stale"],
                "days_old": result["days_old"],
            }
            if "arrival_qty" in result:
                response["arrival_qty"] = result["arrival_qty"]
                response["arrival_unit"] = result["arrival_unit"]
            await params.result_callback(response)
        else:
            await params.result_callback({
                "error": (
                    f"No {commodity_name} price data found for {district!r} in {resolved_state} within the "
                    f"last {_MAX_STALE_DAYS} days — "
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


# Same cap as location_lookup.py's _MAX_CANDIDATES_TO_OFFER — above this many
# markets, don't read the whole list by default, summarize instead.
_MAX_MARKETS_TO_LIST = 8


def make_get_price_all_markets(serializer=None):
    """Builds the get_price_all_markets direct-function tool — the
    no-district counterpart to get_price, for when the caller asks a
    commodity's price without naming (or being able to name) a district,
    e.g. "onion rate ఎంత" with nothing else. Same ≤8-list / >8-summarize
    shaping as lookup_place_by_pincode's too_many_to_list branch."""

    # cancel_on_interruption=False — same reasoning as make_get_price's
    # identical decorator: a cancelled-mid-flight lookup must never surface
    # to the LLM as a fake "CANCELLED" result it might mistake for "no data".
    @tool_options(cancel_on_interruption=False)
    async def get_price_all_markets(params: FunctionCallParams, commodity: str, state: str) -> None:
        """Get today's price for a commodity across every market yard in a
        state that currently has data for it — use this when the caller
        asks a commodity's price but hasn't named (or can't name) a
        specific district/town, instead of asking them to pick one blind.

        Only call this once you know the commodity and the state — if the
        caller hasn't given a state yet and you can't infer it, ask first.
        If the caller later names a specific district, prefer get_price
        instead of this one.

        Args:
            commodity: The crop, vegetable, fruit, spice, or other
                commodity the caller asked about, in English or as close to
                their own words as possible (for example "onion", "cotton",
                "cashew"). Never guess a commodity the caller didn't name.
            state: The Indian state or union territory the caller is
                calling from (for example "Andhra Pradesh", "Maharashtra").
        """
        _t0 = time.monotonic()

        crop_match = _resolve_commodity(commodity)
        if crop_match is None:
            await params.result_callback({
                "error": f"{commodity!r} isn't a commodity Agmarknet tracks — ask the caller to name it differently.",
            })
            return
        crop_keyword, _group, commodity_name = crop_match

        resolved_state = _resolve_state(state)
        if resolved_state is None:
            await params.result_callback({
                "error": f"{state!r} isn't a recognized Indian state/UT — ask the caller to confirm which state.",
            })
            return

        rows = await fetch_price_all_markets(crop_keyword, resolved_state)
        exec_ms = int((time.monotonic() - _t0) * 1000)

        if not rows:
            await params.result_callback({
                "error": (
                    f"No {commodity_name} price data found anywhere in {resolved_state} within the last "
                    f"{_MAX_STALE_DAYS} days — this is a gap in the government mandi data."
                ),
            })
            outcome = "no data found"
        else:
            cheapest, costliest = rows[0], rows[-1]
            average_per_kg = round(sum(r["modal_per_kg"] for r in rows) / len(rows), 2)
            summary = {
                "commodity": commodity_name,
                "state": resolved_state,
                "market_count": len(rows),
                # arrival_date/stale were missing here originally — confirmed
                # live (call_919390427476_5243c28b.log, 2026-08-07): the bot
                # read out ₹18.33/kg and ₹28/kg for two markets that were
                # BOTH stale=True with no date or "this is a bit old"
                # caveat at all, because it had no arrival_date anywhere in
                # this response to work from (unlike get_price, which always
                # included one) — there was nothing for it to say.
                "cheapest": {
                    "district": cheapest["district"], "market": cheapest["market"],
                    "modal_per_kg": cheapest["modal_per_kg"],
                    "arrival_date": cheapest["arrival_date"], "stale": cheapest["stale"],
                    "days_old": cheapest["days_old"],
                },
                "costliest": {
                    "district": costliest["district"], "market": costliest["market"],
                    "modal_per_kg": costliest["modal_per_kg"],
                    "arrival_date": costliest["arrival_date"], "stale": costliest["stale"],
                    "days_old": costliest["days_old"],
                },
                "average_per_kg": average_per_kg,
            }
            if len(rows) <= _MAX_MARKETS_TO_LIST:
                summary["markets"] = [
                    {
                        "district": r["district"], "market": r["market"],
                        "modal_per_kg": r["modal_per_kg"],
                        "arrival_date": r["arrival_date"], "stale": r["stale"],
                        "days_old": r["days_old"],
                    }
                    for r in rows
                ]
            else:
                # Too many to read by default — min/max/average above already
                # answers "what's onion going for", full breakdown only on
                # explicit request (same all_villages pattern as
                # lookup_place_by_pincode's too_many_to_list branch).
                summary["too_many_to_list"] = True
                summary["all_markets"] = [
                    {
                        "district": r["district"], "market": r["market"],
                        "modal_per_kg": r["modal_per_kg"],
                        "arrival_date": r["arrival_date"], "stale": r["stale"],
                        "days_old": r["days_old"],
                    }
                    for r in rows
                ]
            await params.result_callback(summary)
            outcome = f"{len(rows)} markets, ₹{cheapest['modal_per_kg']:.2f}-₹{costliest['modal_per_kg']:.2f}/kg"

        if serializer is not None and serializer.call_id:
            track_task(log_tool_call(
                serializer.call_id, "market_price_lookup_all",
                f"{commodity} across {state}", outcome, bool(rows), exec_ms,
            ))

    return get_price_all_markets
