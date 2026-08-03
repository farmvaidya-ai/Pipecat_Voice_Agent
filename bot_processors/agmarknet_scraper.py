"""Drives the real agmarknet.gov.in "Daily Price and Arrival Report" page
with a real (headless) browser — filling the same State/Commodity
Group/Commodity dropdowns and clicking the same "Go" button a human would —
and reads today's prices out of the rendered results table.

scrape_all() is the background job: it's meant to be run periodically
(bot_processors/scraper_daemon.py) and writes straight into
bot_processors/last_known_prices.json — the same cache file
bot_processors.price_lookup.fetch_price() reads from — so calls always see
whatever this job most recently scraped. Nothing about what it scrapes is
hand-picked: every state/UT (via the "All States/UTs" option) and every
commodity (from load_commodity_reference(), harvested from the site's own
dropdowns) is walked dynamically.

scrape_single() is the live counterpart bot_processors.price_lookup calls
mid-call for a cache miss (e.g. right after this file's own commodity
reference was refreshed, before the next scrape_all() run has caught up):
driving the same multi-step cascading form still takes several seconds, so
this is only ever used as a one-time fetch for a combo not yet cached, with
the result persisted afterward so subsequent callers get the cached fast
path and scrape_all() keeps it fresh from then on.

Run manually with:  python -m bot_processors.agmarknet_scraper
"""

import base64
import json
import time
from datetime import date
from pathlib import Path

from loguru import logger
from playwright.sync_api import Locator, Page, sync_playwright

from bot_processors.price_shared import _LAST_KNOWN_KEY_SEP, _LAST_KNOWN_PATH, crop_keyword_for

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# Agmarknet started requiring a captcha on every "Go" submission (as of
# 2026-07-31 — last_known_prices.json has real data up to 30-07-2026, then
# every scrape that morning silently returned 0 rows because the query never
# actually ran, rejected server-side for a missing/empty captcha field).
# Solved with OCR rather than any external captcha-solving service, since the
# image is a clean, low-noise 6-character alphanumeric render with no CAPTCHA
# challenge beyond that. _ocr_reader is a lazy singleton because loading the
# model is the expensive part (~seconds) and every Go click in a long run
# reuses the same one.
_CAPTCHA_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
_ocr_reader = None

# The site shows this full-screen loading overlay after most dropdown/date
# changes while it fetches the next field's options. If the Go click lands
# while it's still up, Playwright's own actionability retry hammers the
# click for its default 30s timeout before giving up — that's most of the
# per-commodity time seen in production, not the captcha. Waiting for it to
# detach first resolves in ~1-2s in the normal case; the explicit short
# timeout on the click itself is only a fallback for the rare case where the
# overlay never clears, so a stuck query fails in seconds instead of 30s and
# hands off to scrape_all's existing per-commodity recovery sooner.
_OVERLAY_SELECTOR = "div.fixed.inset-0.bg-black.bg-opacity-30.backdrop-blur-sm"
_GO_CLICK_TIMEOUT_MS = 15000


def _wait_overlay_gone(page: Page, timeout: int = 5000) -> None:
    try:
        page.locator(_OVERLAY_SELECTOR).wait_for(state="hidden", timeout=timeout)
    except Exception:
        pass


def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr

        _ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _ocr_reader


def _solve_captcha(page: Page) -> None:
    """Reads the captcha's base64 image data straight off the <img> element
    (no extra network round-trip) and fills its OCR reading into the
    #captcha field. Not reliable on its own — see _click_go, which is the
    real entry point and retries this against a freshly refreshed image on
    rejection, since ambiguous glyphs (I/l/1, 0/O/9/g) mean a single OCR
    pass gets it wrong more often than not."""
    src = page.locator("img[alt='Captcha']").get_attribute("src")
    _, b64data = src.split(",", 1)
    image_bytes = base64.b64decode(b64data)

    reader = _get_ocr_reader()
    guess = "".join(reader.readtext(image_bytes, detail=0, allowlist=_CAPTCHA_ALPHABET))
    page.fill("#captcha", guess)


def _click_go(page: Page, max_attempts: int = 12) -> None:
    """Solves the captcha and clicks Go, retrying against a freshly
    refreshed captcha image whenever the site rejects the guess. The
    #captcha input's own aria-invalid attribute is the accept/reject signal
    — flips to "true" on a rejected submission and back to "false" once one
    is accepted — confirmed more reliable than scraping for an error message
    (observed via real testing: no visible error text reappears on repeat
    rejections, only this attribute changes). OCR gets this captcha's
    ambiguous 6-character glyphs right well under half the time, so
    max_attempts is generously sized rather than tuned tight — a few extra
    ~3s retries is cheap next to a query silently returning nothing."""
    for attempt in range(max_attempts):
        _solve_captcha(page)
        _wait_overlay_gone(page)
        page.click("button:has-text('Go')", timeout=_GO_CLICK_TIMEOUT_MS)

        # Rejection (aria-invalid flips to "true") and an accepted-with-rows
        # response (a result row renders) usually resolve well under 2s —
        # polling for either lets a fast response return early instead of
        # always paying the full fixed wait. An accepted-but-genuinely-empty
        # response has no earlier DOM signal to poll for, so it still falls
        # back to the full 2000ms in that case, same as before.
        try:
            page.wait_for_function(
                "() => document.querySelector('#captcha')?.getAttribute('aria-invalid') === 'true' "
                "|| document.querySelectorAll('table tbody tr').length > 0",
                timeout=2000,
            )
        except Exception:
            pass

        if page.locator("#captcha").get_attribute("aria-invalid") == "false":
            return

        logger.warning(
            f"⚠️ agmarknet_scraper: captcha rejected (attempt {attempt + 1}/{max_attempts}), refreshing and retrying"
        )
        page.click("button[aria-label='Refresh captcha']")
        page.wait_for_timeout(800)

    logger.error(f"⚠️ agmarknet_scraper: captcha solving failed after {max_attempts} attempts, giving up on this query")


def _open_dropdown(page: Page, label_text: str) -> Locator:
    """Opens a cascading search-select dropdown by its label and returns its
    trigger element, retrying until the search box actually appears.

    Re-opening a field already picked earlier in the same page session (e.g.
    switching State/UT from one state to the next, or Commodity Group from
    one group to the next) sometimes toggles an already-open/stale panel
    closed instead of opening it, leaving no search box on screen — this is
    the retry this function exists for. Shared by _pick() (select one named
    option) and _enumerate_dropdown_options() (read every option without
    selecting any), since both need the panel open first.
    """
    trigger = page.locator(f"label:text-is('{label_text}')").locator("..")
    search_box = page.locator("input[placeholder='Search...']")

    for _ in range(5):
        trigger.click(force=True)
        page.wait_for_timeout(500)
        if search_box.count() > 0 and search_box.last.is_visible():
            return trigger
    raise RuntimeError(f"agmarknet_scraper: dropdown for {label_text!r} never opened")


def _enumerate_dropdown_options(page: Page, trigger: Locator) -> list[str]:
    """Reads every option currently listed in an already-open dropdown panel
    (see _open_dropdown), without selecting any of them. The panel is a
    sibling of the trigger's own parent, not a descendant of the trigger
    itself — confirmed by inspecting the live site's rendered DOM."""
    outer = trigger.locator("..")
    panel = outer.locator("div.absolute.top-full")
    return panel.locator("div.py-1 > div").all_inner_texts()


def _pick(page: Page, label_text: str, option_text: str) -> None:
    """Opens a cascading search-select dropdown by its label, types the
    option text into its search box, and clicks the matching option — same
    interaction a caller filling the form by hand would perform.

    Beyond the dropdown-never-opened failure _open_dropdown retries for,
    a second independent flaky failure mode was found here, also silent (no
    exception, no visible error) — a scraper run would just keep querying
    the previous field's value: even once the search box is open and the
    option is clicked, the click occasionally doesn't register with the
    underlying React state (observed: trigger briefly showed both the old
    and new value concatenated, e.g. "Andhra Pradesh Telang...", then
    settled back on the OLD value) — so the form silently kept submitting
    the previous selection. This produced 41 real cache entries scraped
    under a "Telangana" key that were actually re-scraped Andhra Pradesh
    rows. Fixed by reading the trigger's displayed text back after clicking
    and retrying the whole open/search/click sequence if it doesn't contain
    the value we just tried to select.
    """
    search_box = page.locator("input[placeholder='Search...']")

    for attempt in range(3):
        trigger = _open_dropdown(page, label_text)

        box = search_box.last
        box.fill(option_text)
        page.wait_for_timeout(800)
        page.locator(f"text='{option_text}'").last.click(force=True)
        page.wait_for_timeout(600)

        if option_text in trigger.inner_text():
            return
        logger.warning(
            f"⚠️ agmarknet_scraper: {label_text!r} still doesn't show "
            f"{option_text!r} after selecting it (attempt {attempt + 1}/3), retrying"
        )

    raise RuntimeError(
        f"agmarknet_scraper: failed to select {option_text!r} for {label_text!r} after 3 attempts"
    )


def _scrape_visible_results_dynamic(page: Page) -> list[dict[str, str]]:
    """Reads every row of the results table, paging through with the
    "Next page" button (the site only ever offers 10 rows/page, no larger
    page-size option) until it's disabled or absent. Reads column names from
    the table's own header row rather than assuming a fixed layout — needed
    for "Both" mode. The thead there has two rows: a grouping row
    ("COMMODITY INFO" / "PRICE INFO" / "ARRIVAL INFO", 3 cells) above the
    real per-column header row — only the last thead row's cell count
    matches each body row's <td> count, so that's the one to read."""
    rows: list[dict[str, str]] = []
    table = page.locator("table")
    if table.count() == 0:
        return rows
    headers = [
        h.strip() or f"col_{i}"
        for i, h in enumerate(table.locator("thead tr").last.locator("th").all_inner_texts())
    ]

    while True:
        trs = table.locator("tbody tr")
        for i in range(trs.count()):
            cells = trs.nth(i).locator("td").all_inner_texts()
            if len(cells) == len(headers):
                rows.append(dict(zip(headers, (c.strip() for c in cells))))

        next_btn = page.locator('button[title="Next page"]')
        if next_btn.count() == 0 or next_btn.get_attribute("disabled") is not None:
            break
        next_btn.click()
        page.wait_for_timeout(1000)
    return rows


def _row_to_result(row: dict[str, str]) -> dict | None:
    """Converts one scraped table row into a last_known_prices.json entry.
    scrape_all()/scrape_single() drive the form in "Both" mode (see
    _open_daily_price_form), so `row` comes from the dynamic header-driven
    reader (_scrape_visible_results_dynamic) and its keys are the site's own
    column labels ("State/UT", "Min Price", ...).
    Arrival Quantity/Unit are only added to the result when the site
    actually reported a number for them — "NR" ("not reported") and blank
    are both common and shouldn't produce a fake "0" arrival figure the bot
    could speak as if it were real."""
    unit = row["Price Unit"]
    try:
        min_price = float(row["Min Price"].replace(",", ""))
        max_price = float(row["Max Price"].replace(",", ""))
        modal_price = float(row["Modal Price"].replace(",", ""))
    except ValueError:
        return None

    if "quintal" in unit.lower():
        divisor = 100.0
    elif "kg" in unit.lower():
        divisor = 1.0
    else:
        logger.warning(f"⚠️ agmarknet_scraper: unrecognized price unit {unit!r}, skipping row")
        return None

    result = {
        "commodity": row["Commodity"],
        "market": row["Market"],
        "district": row["District"],
        "state": row["State/UT"],
        "arrival_date": row["Price Date"],
        "modal_per_kg": round(modal_price / divisor, 2),
        "min_per_kg": round(min_price / divisor, 2),
        "max_per_kg": round(max_price / divisor, 2),
        "stale": False,
    }

    arrival_qty = row.get("Arrival Quantity", "").strip()
    arrival_unit = row.get("Arrival Unit", "").strip()
    if arrival_qty and arrival_qty.upper() != "NR":
        result["arrival_qty"] = arrival_qty
        result["arrival_unit"] = arrival_unit

    return result


def _open_daily_price_form(page: Page, mode: str = "Price") -> None:
    """Navigation shared by every entry point that drives the live form:
    scrape_all(), scrape_single(), and harvest_commodity_reference(). mode
    is one of the "Price/Arrivals*" dropdown's own three options: "Price",
    "Arrival", or "Both" (a combined table with both price and arrival
    columns per row)."""
    page.goto("https://agmarknet.gov.in/home", wait_until="networkidle", timeout=60000)
    page.wait_for_timeout(1500)
    page.click("text=Price & Arrival Reports")
    page.wait_for_timeout(1500)
    page.click("text=Daily Price and Arrival")
    page.wait_for_timeout(3000)
    _pick(page, "Price/Arrivals*", mode)


def _scrape_state_crop(page: Page, state: str, group: str, commodity: str, crop_keyword: str) -> dict[str, dict]:
    """Runs one (state, commodity) query against the already-open,
    already-state-picked form and returns a "state||district||crop_keyword"
    -> result dict for whichever districts had rows. Shared by scrape_all()
    (loops this over every target) and scrape_single() (calls it once)."""
    _pick(page, "Commodity Group*", group)
    _pick(page, "Commodity*", commodity)
    page.keyboard.press("Escape")
    page.wait_for_timeout(400)
    _click_go(page)
    page.wait_for_timeout(500)

    rows = _scrape_visible_results_dynamic(page)

    # An empty result is sometimes a genuine "nothing reported today" (e.g.
    # paddy/soyabean are out of season here in July) and sometimes just a
    # transient site hiccup around the Go click — this exact (state, crop)
    # combo has been observed to return 0 rows on one run and 10+ on the
    # next with no code change in between. Re-clicking Go once (form fields
    # are already correctly set, no need to re-pick them) filters out the
    # transient case cheaply before accepting an empty result as real.
    if not rows:
        page.wait_for_timeout(1500)
        _click_go(page)
        page.wait_for_timeout(500)
        rows = _scrape_visible_results_dynamic(page)

    # Even with the State/UT picker now verified (see _pick), the results
    # table itself has been observed to briefly carry over one stale row
    # from the previous state's query after clicking Go — a render race,
    # not a picker failure. Trust the row's own scraped "State/UT" cell over
    # the state we intended to query, and drop anything that disagrees
    # rather than let a stray row corrupt this state's cache entries (this
    # is exactly how 41 Andhra Pradesh rows ended up saved under
    # "Telangana" keys before this fix).
    stray = [row for row in rows if row["State/UT"] != state]
    if stray:
        logger.warning(
            f"⚠️ agmarknet_scraper: dropping {len(stray)} stale row(s) "
            f"from a different state while querying {state}/{commodity}"
        )
    rows = [row for row in rows if row["State/UT"] == state]

    logger.info(f"🌾 agmarknet_scraper: {state}/{commodity} -> {len(rows)} rows")

    scraped: dict[str, dict] = {}
    for row in rows:
        result = _row_to_result(row)
        if result is None:
            continue
        key = _LAST_KNOWN_KEY_SEP.join((state, result["district"], crop_keyword))
        scraped.setdefault(key, result)
    return scraped


def _scrape_crop_all_states(page: Page, group: str, commodity: str, crop_keyword: str) -> dict[str, dict]:
    """Runs one commodity query against the already-open form with
    State/UT fixed to "All States/UTs" (picked once by scrape_all() before
    looping crops), so a single query returns every state/UT's rows for
    this commodity at once — replaces the old per-state loop entirely,
    since there's no longer one specific state to filter stray rows
    against. Keys each row by its own scraped state, not a requested one."""
    _pick(page, "Commodity Group*", group)
    _pick(page, "Commodity*", commodity)
    page.keyboard.press("Escape")
    page.wait_for_timeout(400)
    _click_go(page)
    page.wait_for_timeout(500)

    rows = _scrape_visible_results_dynamic(page)
    if not rows:
        page.wait_for_timeout(1500)
        _click_go(page)
        page.wait_for_timeout(500)
        rows = _scrape_visible_results_dynamic(page)

    logger.info(f"🌾 agmarknet_scraper: {commodity} (all states) -> {len(rows)} rows")

    scraped: dict[str, dict] = {}
    for row in rows:
        result = _row_to_result(row)
        if result is None:
            continue
        key = _LAST_KNOWN_KEY_SEP.join((result["state"], result["district"], crop_keyword))
        scraped.setdefault(key, result)
    return scraped


def scrape_all(resume: bool = False) -> dict[str, dict]:
    """Runs every commodity Agmarknet tracks (~604, from
    load_commodity_reference() — not a hand-picked list) against every
    state/UT at once (via the "All States/UTs" option, one query per
    commodity rather than one per state x commodity) and returns a
    "state||district||crop_keyword" -> result dict of everything scraped
    this call. crop_keyword is derived the same way bot_processors.price_lookup
    derives it when resolving a caller's spoken commodity (see
    price_shared.crop_keyword_for) so a live call's lookup key always
    matches what this just wrote. One row per key is kept (first seen
    wins) — good enough for an approximate fallback price, not exact
    market analytics.

    Drives the form in "Both" mode (price + arrival columns in one query)
    so each cached entry also gets arrival_qty/arrival_unit alongside the
    existing per-kg price fields, when the site reported one — see
    _row_to_result.

    Persists to last_known_prices.json after each commodity (via
    save_scraped), not just once at the end — a full run is ~604 form
    submissions and can take hours, and a mid-run crash used to lose the
    entire run's progress since nothing was written until scrape_all()
    returned. A single commodity's dropdown pick failing is logged and
    skipped (form reopened and re-picked to recover) rather than aborting
    the whole run.

    resume=True skips any crop_keyword that already has at least one entry
    dated today in last_known_prices.json — for relaunching after a crash
    without re-querying commodities this run already got to today. Not the
    default: the scheduled daemon (scraper_daemon.py) wants every cycle to
    be a full unconditional refresh, not skip a commodity just because an
    earlier cycle today already touched it."""
    targets: dict[str, tuple[str, str]] = {}
    for group, commodities in load_commodity_reference().items():
        for commodity in commodities:
            targets[crop_keyword_for(commodity)] = (group, commodity)

    today = date.today().strftime("%d-%m-%Y")
    done_today: set[str] = set()
    if resume and _LAST_KNOWN_PATH.exists():
        existing = json.loads(_LAST_KNOWN_PATH.read_text(encoding="utf-8"))
        for key, entry in existing.items():
            if entry.get("arrival_date") == today:
                done_today.add(key.split(_LAST_KNOWN_KEY_SEP)[-1])
        if done_today:
            logger.info(
                f"⏭️ agmarknet_scraper: resuming — {len(done_today)}/{len(targets)} "
                "commodities already have today's data"
            )

    scraped: dict[str, dict] = {}
    done_count = len(done_today)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--ignore-certificate-errors"])
        page = browser.new_page(
            viewport={"width": 1400, "height": 1000},
            ignore_https_errors=True,
            user_agent=_USER_AGENT,
        )
        _open_daily_price_form(page, mode="Both")
        _pick(page, "State/UT*", "All States/UTs")

        for crop_keyword, (group, commodity) in targets.items():
            if crop_keyword in done_today:
                continue

            try:
                commodity_scraped = _scrape_crop_all_states(page, group, commodity, crop_keyword)
            except Exception:
                logger.opt(exception=True).warning(
                    f"⚠️ agmarknet_scraper: failed on {group}/{commodity}, reloading form and skipping"
                )
                try:
                    _open_daily_price_form(page, mode="Both")
                    _pick(page, "State/UT*", "All States/UTs")
                except Exception:
                    logger.opt(exception=True).error(
                        "⚠️ agmarknet_scraper: form recovery failed, aborting run"
                    )
                    raise
                continue

            scraped.update(commodity_scraped)
            if commodity_scraped:
                save_scraped(commodity_scraped)
            done_count += 1
            logger.info(f"📊 agmarknet_scraper: {done_count}/{len(targets)} commodities done")
            time.sleep(1.5)  # be polite between form submissions

        browser.close()

    return scraped


def scrape_single(state: str, group: str, commodity: str, crop_keyword: str) -> dict[str, dict]:
    """Live, on-demand equivalent of one (state, crop) iteration of
    scrape_all() — launches its own short-lived browser session so it can
    run mid-call without waiting for or interfering with the scheduled
    background job. District stays "All Districts" same as scrape_all(), so
    one call still covers every district in the state, not just the
    caller's own. Takes several seconds (browser launch + one form
    submission) — meant to be awaited from an async context via
    asyncio.to_thread, not called directly from an event loop."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--ignore-certificate-errors"])
        page = browser.new_page(
            viewport={"width": 1400, "height": 1000},
            ignore_https_errors=True,
            user_agent=_USER_AGENT,
        )
        _open_daily_price_form(page, mode="Both")
        _pick(page, "State/UT*", state)
        result = _scrape_state_crop(page, state, group, commodity, crop_keyword)
        browser.close()
    return result


# Full Agmarknet commodity taxonomy (group -> [commodity, ...]), harvested
# once from the site's own Commodity Group/Commodity dropdowns rather than
# hand-picked — see harvest_commodity_reference(). Used by
# bot_processors.price_lookup to recognize a commodity outside the
# original 13 when the caller names it in English or a common loanword.
_COMMODITY_REFERENCE_PATH = Path(__file__).parent / "agmarknet_commodities.json"


def harvest_commodity_reference() -> dict[str, list[str]]:
    """Walks every Commodity Group and, for each, every Commodity inside it,
    reading the dropdown's own rendered option list rather than assuming any
    fixed set — so this stays correct if Agmarknet adds/renames commodities,
    without another round of manual hardcoding. Takes roughly a minute (16
    groups today); meant to be run occasionally (see scraper_daemon.py),
    not on every scrape."""
    reference: dict[str, list[str]] = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--ignore-certificate-errors"])
        page = browser.new_page(
            viewport={"width": 1400, "height": 1000},
            ignore_https_errors=True,
            user_agent=_USER_AGENT,
        )
        _open_daily_price_form(page)

        group_trigger = _open_dropdown(page, "Commodity Group*")
        groups = _enumerate_dropdown_options(page, group_trigger)
        page.keyboard.press("Escape")
        page.wait_for_timeout(300)

        for group in groups:
            _pick(page, "Commodity Group*", group)
            commodity_trigger = _open_dropdown(page, "Commodity*")
            commodities = _enumerate_dropdown_options(page, commodity_trigger)
            page.keyboard.press("Escape")
            page.wait_for_timeout(300)
            reference[group] = commodities
            logger.info(f"📚 agmarknet_scraper: {group} -> {len(commodities)} commodities")

        browser.close()

    _COMMODITY_REFERENCE_PATH.write_text(
        json.dumps(reference, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(
        f"💾 agmarknet_scraper: wrote {sum(len(v) for v in reference.values())} "
        f"commodities across {len(reference)} groups to {_COMMODITY_REFERENCE_PATH}"
    )
    return reference


def load_commodity_reference() -> dict[str, list[str]]:
    if not _COMMODITY_REFERENCE_PATH.exists():
        return {}
    return json.loads(_COMMODITY_REFERENCE_PATH.read_text(encoding="utf-8"))


# Full list of State/UT options Agmarknet's own dropdown offers, harvested
# once rather than hand-typed — see harvest_state_reference(). Unlike the
# commodity reference this is small and effectively permanent (India's
# states/UTs don't change often), but harvesting it the same way keeps it
# exactly in sync with the site's own spelling (the form only accepts one
# of these exact strings). Used by bot_processors.price_lookup to resolve
# whatever state name a caller (or the LLM inferring one) gives, without a
# hand-maintained alias table.
_STATE_REFERENCE_PATH = Path(__file__).parent / "agmarknet_states.json"


def harvest_state_reference() -> list[str]:
    """Walks the State/UT dropdown's own rendered option list once. Takes
    well under a minute — one dropdown, not a cascading walk like
    harvest_commodity_reference()."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--ignore-certificate-errors"])
        page = browser.new_page(
            viewport={"width": 1400, "height": 1000},
            ignore_https_errors=True,
            user_agent=_USER_AGENT,
        )
        _open_daily_price_form(page)
        state_trigger = _open_dropdown(page, "State/UT*")
        states = _enumerate_dropdown_options(page, state_trigger)
        page.keyboard.press("Escape")
        browser.close()

    _STATE_REFERENCE_PATH.write_text(
        json.dumps(states, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(f"💾 agmarknet_scraper: wrote {len(states)} states/UTs to {_STATE_REFERENCE_PATH}")
    return states


def load_state_reference() -> list[str]:
    if not _STATE_REFERENCE_PATH.exists():
        return []
    return json.loads(_STATE_REFERENCE_PATH.read_text(encoding="utf-8"))


def save_scraped(scraped: dict[str, dict]) -> None:
    """Merges freshly scraped results into last_known_prices.json, keeping
    any existing keys this run didn't touch (e.g. a crop/state combo that
    came back empty today) rather than discarding them."""
    existing: dict[str, dict] = {}
    if _LAST_KNOWN_PATH.exists():
        existing = json.loads(_LAST_KNOWN_PATH.read_text(encoding="utf-8"))

    existing.update(scraped)
    _LAST_KNOWN_PATH.write_text(
        json.dumps(dict(sorted(existing.items())), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"💾 agmarknet_scraper: wrote {len(scraped)} fresh entries to {_LAST_KNOWN_PATH}")


if __name__ == "__main__":
    import sys

    if "--harvest-commodities" in sys.argv:
        harvest_commodity_reference()
    elif "--harvest-states" in sys.argv:
        harvest_state_reference()
    else:
        save_scraped(scrape_all(resume="--resume" in sys.argv))
