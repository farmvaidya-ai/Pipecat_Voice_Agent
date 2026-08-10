"""PostgreSQL-backed caller registry: tracks which phone numbers have called
before and how many times. Separate from caller_memory.py, which stores the
actual per-call summary text — this only tracks caller metadata
(first_seen/last_seen/call_count). Uses the shared pool from db_pool.py —
the `dim_contacts` table itself lives in the same Postgres database as the
voice_agent_db.py tables (see db/schema.sql).
"""

from loguru import logger

from bot_processors.core.db_pool import get_pool


async def record_call(phone_number: str) -> int:
    """Upserts this caller's row and returns their call_count including this
    call (0 on failure or if phone_number is blank)."""
    if not phone_number:
        return 0
    try:
        pool = get_pool()
        async with pool.acquire() as conn:
            count = await conn.fetchval(
                """
                INSERT INTO dim_contacts (phone_number, first_seen, last_seen, call_count)
                VALUES ($1, NOW(), NOW(), 1)
                ON CONFLICT (phone_number) DO UPDATE SET
                    last_seen = EXCLUDED.last_seen,
                    call_count = dim_contacts.call_count + 1
                RETURNING call_count
                """,
                phone_number,
            )
        logger.info(f"📇 caller_db: {phone_number} — call #{count}")
        return count
    except Exception:
        logger.opt(exception=True).warning(f"⚠️ caller_db: failed to record call for {phone_number}")
        return 0


async def get_location(phone_number: str) -> dict | None:
    """This caller's verbally captured farming location (state/district/
    mandal/village/market/pincode), if the metadata-capture flow has ever
    completed for them — see bot_processors/location_lookup.py. None if
    never captured."""
    if not phone_number:
        return None
    try:
        pool = get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT confirmed_state, confirmed_district, confirmed_mandal, confirmed_village, "
                "confirmed_market, confirmed_pincode FROM dim_contacts WHERE phone_number = $1",
                phone_number,
            )
        if row is None or not row["confirmed_district"]:
            return None
        return {
            "state": row["confirmed_state"],
            "district": row["confirmed_district"],
            "mandal": row["confirmed_mandal"],
            "village": row["confirmed_village"],
            "market": row["confirmed_market"],
            "pincode": row["confirmed_pincode"],
        }
    except Exception:
        logger.opt(exception=True).warning(f"⚠️ caller_db: failed to load location for {phone_number}")
        return None


async def save_location(
    phone_number: str, state: str, district: str, mandal: str, village: str, pincode: str, market: str = ""
) -> bool:
    """Persists the caller's verbally captured state/district/mandal/village/
    pincode (dim_contacts row already exists by the time this is called —
    record_call upserts it at call start, before any tool call can happen)
    so later calls from this number can skip the metadata-capture flow
    entirely. market is optional — it's only ever set via the separate
    market-yard confirmation flow, not the plain metadata sequence."""
    if not phone_number or not district:
        return False
    try:
        pool = get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE dim_contacts
                SET confirmed_state = $2, confirmed_district = $3, confirmed_mandal = $4,
                    confirmed_village = $5, confirmed_pincode = $6,
                    confirmed_market = COALESCE($7, confirmed_market), location_confirmed_at = NOW()
                WHERE phone_number = $1
                """,
                phone_number, state, district, mandal or None, village or None, pincode or None,
                market or None,
            )
        logger.info(f"📍 caller_db: saved location for {phone_number} — {village or mandal or district}, {state}")
        return True
    except Exception:
        logger.opt(exception=True).warning(f"⚠️ caller_db: failed to save location for {phone_number}")
        return False


async def update_contact_name(phone_number: str, name: str) -> None:
    """Sets dim_contacts.name once it's known (from caller_memory's saved
    summary) — record_call() runs before that name is available (it's
    fetched concurrently, see Bot.py), so this is a separate follow-up write
    rather than part of record_call()'s upsert."""
    if not phone_number or not name:
        return
    try:
        pool = get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE dim_contacts SET name = $2 WHERE phone_number = $1",
                phone_number, name,
            )
    except Exception:
        logger.opt(exception=True).warning(f"⚠️ caller_db: failed to update name for {phone_number}")
