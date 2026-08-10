"""Per-call structured logging into fact_sessions / fact_conversations /
fact_toolcalls / fact_performance (see db/schema.sql).

Same shared Postgres pool as caller_db.py / caller_memory.py. Every function
here is best-effort: failures are logged and swallowed, never allowed to
affect the live call. All writes should go through bot_processors.
task_tracker.track_task so a slow/failed insert can't add latency or get
silently garbage-collected mid-flight.
"""

from datetime import datetime

from loguru import logger

from bot_processors.core.db_pool import get_pool


async def start_call(call_id: str, phone_number: str, start_time: datetime) -> None:
    if not call_id:
        return
    try:
        pool = get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO fact_sessions (call_id, phone_number, direction, start_time, status)
                VALUES ($1, $2, 'inbound', $3, 'in_progress')
                ON CONFLICT (call_id) DO NOTHING
                """,
                call_id, phone_number or "", start_time,
            )
    except Exception:
        logger.opt(exception=True).warning(f"⚠️ call_db: failed to start call row for {call_id}")


async def end_call(
    call_id: str, end_time: datetime, duration_seconds: int | None, end_reason: str,
    language: str = "",
) -> None:
    if not call_id:
        return
    try:
        pool = get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE fact_sessions
                SET end_time = $2, duration_seconds = $3, status = 'completed', end_reason = $4,
                    language = COALESCE(NULLIF($5, ''), language)
                WHERE call_id = $1
                """,
                call_id, end_time, duration_seconds, end_reason, language,
            )
    except Exception:
        logger.opt(exception=True).warning(f"⚠️ call_db: failed to end call row for {call_id}")


async def save_conversation_messages(
    call_id: str, turns: list[dict], turn_latencies_ms: list[int] | None = None,
) -> None:
    """turn_latencies_ms (see bot_processors/latency.py's _LatencyState) is
    one STT-final->TTS-first-audio measurement per completed assistant turn,
    recorded live during the call in the same order those turns occur here —
    zipped onto the assistant rows below so latency_ms isn't left NULL.
    Only user turns trigger a measurement, so latencies are matched to
    assistant messages specifically; a count mismatch (e.g. a turn cut short
    by an interruption never reaching TTS) just leaves the extra row(s)
    without a latency rather than guessing."""
    if not call_id or not turns:
        return
    latencies = iter(turn_latencies_ms or [])
    rows = [
        (
            call_id,
            "user" if t.get("role") == "user" else "assistant",
            str(t.get("content", "")),
            next(latencies, None) if t.get("role") == "assistant" else None,
        )
        for t in turns
        if t.get("role") in ("user", "assistant") and t.get("content")
    ]
    if not rows:
        return
    try:
        pool = get_pool()
        async with pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO fact_conversations (call_id, speaker, message, latency_ms)
                VALUES ($1, $2, $3, $4)
                """,
                rows,
            )
    except Exception:
        logger.opt(exception=True).warning(f"⚠️ call_db: failed to save conversation messages for {call_id}")


async def log_tool_call(
    call_id: str, tool_name: str, input_text: str, output_text: str,
    success: bool, execution_time_ms: int,
) -> None:
    if not call_id:
        return
    try:
        pool = get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO fact_toolcalls (call_id, tool_name, input, output, success, execution_time_ms)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                call_id, tool_name, input_text, output_text, success, execution_time_ms,
            )
    except Exception:
        logger.opt(exception=True).warning(f"⚠️ call_db: failed to log tool call for {call_id}")


async def log_performance_metric(
    call_id: str,
    stt_ms: int | None,
    llm_ttft_ms: int | None,
    llm_total_ms: int | None,
    tts_ms: int | None,
    total_response_ms: int | None,
) -> None:
    if not call_id:
        return
    try:
        pool = get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO fact_performance
                    (call_id, stt_ms, llm_ttft_ms, llm_total_ms, tts_ms, total_response_ms)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                call_id, stt_ms, llm_ttft_ms, llm_total_ms, tts_ms, total_response_ms,
            )
    except Exception:
        logger.opt(exception=True).warning(f"⚠️ call_db: failed to log performance metric for {call_id}")


async def backfill_llm_total_ms(call_id: str, llm_total_ms: int) -> None:
    """Patches the most recent NULL llm_total_ms row for this call.

    Needed because the fact_performance row is written at TTS-first-audio
    time (see latency.py), but for multi-sentence LLM responses TTS starts
    speaking sentence 1 before LLMFullResponseEndFrame fires for the rest —
    so llm_total_ms isn't known yet when that row is first inserted.
    """
    if not call_id:
        return
    try:
        pool = get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE fact_performance SET llm_total_ms = $2
                WHERE id = (
                    SELECT id FROM fact_performance
                    WHERE call_id = $1 AND llm_total_ms IS NULL
                    ORDER BY timestamp DESC LIMIT 1
                )
                """,
                call_id, llm_total_ms,
            )
    except Exception:
        logger.opt(exception=True).warning(f"⚠️ call_db: failed to backfill llm_total_ms for {call_id}")
