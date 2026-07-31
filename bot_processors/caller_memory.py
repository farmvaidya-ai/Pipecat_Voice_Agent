"""Per-caller conversation memory: recognizes returning callers by phone
number (captured from Tata Smartflo's 'start' event, see serializer.py) and
persists a running memory of their calls so the next call from that number
isn't a blank slate.

Storage: the fact_conversation_summary table (see db/schema.sql), one row
per call, in the same Postgres database/pool as caller_db.py and
voice_agent_db.py.
Each saved summary is *cumulative* — caller_summarizer.py folds the previous
summary together with the latest call's transcript into one updated ~300
word summary — so the most recent row alone already carries the caller's
full history forward. That's why loading a caller's context only ever needs
their single latest row, not every row: call 11 loads call 10's summary, and
call 10's summary already absorbed calls 1-9.
"""

from loguru import logger

from bot_processors.db_pool import get_pool


async def load_summaries(caller_number: str) -> list[dict]:
    """This caller's saved calls, oldest first. [] if none/unknown."""
    if not caller_number:
        return []
    try:
        pool = get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT call_id, created_at, name, summary, last_topic
                FROM fact_conversation_summary
                WHERE phone_number = $1
                ORDER BY created_at ASC
                """,
                caller_number,
            )
        return [
            {
                "call_id": row["call_id"],
                "timestamp": row["created_at"].isoformat(),
                "name": row["name"] or "",
                "summary": row["summary"],
                "last_topic": row["last_topic"] or "",
            }
            for row in rows
        ]
    except Exception:
        logger.opt(exception=True).warning(f"⚠️ caller_memory: failed to load summaries for {caller_number}")
        return []


async def load_latest_summary(caller_number: str) -> dict | None:
    """This caller's most recent saved call — already a cumulative merge of
    every call before it (see caller_summarizer.py). None if never called
    before."""
    calls = await load_summaries(caller_number)
    return calls[-1] if calls else None


async def save_summary(
    caller_number: str, summary: str, call_id: str = "", name: str = "", last_topic: str = "",
) -> None:
    """Save this call's (cumulative) summary as a new row under the
    caller's phone number."""
    if not caller_number or not summary:
        return
    try:
        pool = get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO fact_conversation_summary (phone_number, call_id, name, summary, last_topic)
                VALUES ($1, $2, $3, $4, $5)
                """,
                caller_number, call_id or None, name or None, summary, last_topic or None,
            )
        logger.info(f"💾 caller_memory: saved summary for {caller_number} ({name or 'name unknown'}) → call_id={call_id or 'n/a'}")
    except Exception:
        logger.opt(exception=True).warning(f"⚠️ caller_memory: failed to save summary for {caller_number}")


def build_template_greeting(name: str) -> str:
    """Fast, deterministic opening line for a known returning caller — no LLM
    round-trip, so it adds no latency beyond TTS itself (unlike
    caller_summarizer.build_llm_greeting). Doesn't restate the specific past
    topic (that would mean reading the stored summary aloud); just
    acknowledges there was one and asks how to help today."""
    return (
        f"{name} గారు, మిమ్మల్ని మళ్ళీ కలవడం సంతోషంగా ఉంది. "
        "గత సారి మనం మాట్లాడుకున్న విషయం ఎలా ఉంది? ఈరోజు మీకు ఎలా సహాయపడగలను?"
    )


def note_from_summary(latest: dict) -> str:
    """The system-message note text for an already-loaded summary dict (see
    load_latest_summary) — split out from format_context_note so callers
    that already have `latest` in hand (e.g. Bot.py deciding the opening
    greeting) don't have to re-read it from the database just to build the
    note."""
    name = latest.get("name", "")
    name_line = f"This caller's name is {name}. " if name else "This caller has called before. "
    return (
        f"{name_line}Here is what you know about them from past calls, for "
        "your context only — do not read this out loud verbatim:\n"
        + latest.get("summary", "")
    )


async def format_context_note(caller_number: str) -> str | None:
    """A short system-message note for a returning caller, built from their
    single latest (cumulative) summary. None if no history exists."""
    latest = await load_latest_summary(caller_number)
    return note_from_summary(latest) if latest else None
