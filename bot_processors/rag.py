"""RAG retrieval: injects chonkie/Qdrant knowledge-base passages into the LLM context."""

import asyncio
import time

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from bot_processors.call_db import log_tool_call
from bot_processors.intent_router import classify_intent
from bot_processors.task_tracker import track_task
from chonkie_rag.search import search as rag_search

_RAG_MSG_PREFIX = "Relevant knowledge base passages:"
_RAG_PASSAGE_PREFIX = f"{_RAG_MSG_PREFIX}\n"
_RAG_EMPTY_MARKER = (
    f"{_RAG_MSG_PREFIX} none found for this question. This only means the "
    "knowledge base has nothing on it — if the caller is asking about a "
    "mandi/market price or the weather and you already have enough "
    "information (commodity+state+district, or a location), still call "
    "get_price or get_weather; this marker is not a reason to skip that. "
    "Otherwise, stay in character as the Agri Phero Solutionz representative "
    "— do not answer as an AI/language model or mention your own training or "
    "creators. If this question is about the company (e.g. its founders, "
    "history), use the out-of-scope response since that is not in the "
    "knowledge base."
)


class RAGInjector(FrameProcessor):
    """
    Sits before context_aggregator.user(). Retrieves relevant chunks from the
    chonkie/Qdrant knowledge base and injects them into the shared LLMContext
    as a system message, so the LLM sees them right before this turn's user
    question.

    Speculative lookup: a search is fired in the background on each
    InterimTranscriptionFrame (while the user is still talking) instead of
    waiting for the turn to end, so the Qdrant round-trip overlaps with
    speech instead of adding to the STT->LLM gap. The final TranscriptionFrame
    just awaits whatever the latest speculative search returns; only if no
    speculative search was ever fired for this turn does it fall back to a
    fresh synchronous search.

    Only the current turn's passages are kept in context — any RAG system
    message injected by a previous turn is removed first, otherwise every
    turn's passages accumulate forever and stale/irrelevant chunks from
    earlier questions leak into later, unrelated answers.

    STT sometimes splits one continuous utterance into multiple
    TranscriptionFrames (e.g. a mid-sentence pause), all arriving before the
    turn-stop strategy actually triggers LLM inference. Each fragment is
    still searched independently here — but if an earlier fragment's search
    found the right chunk and a later fragment (read in isolation) has none
    of the relevant keywords, a naive per-fragment search+replace would wipe
    out the earlier fragment's correct result before the LLM ever sees it.
    To avoid that, fragments are accumulated into a running turn buffer,
    reset via reset_turn() — called from Bot.py's on_user_turn_started
    handler — and every search runs against the full accumulated text, not
    just the latest fragment.

    The buffer used to be reset by checking whether the last non-system
    message in the shared LLMContext was the assistant's reply — but that
    context mutation can lag behind (the assistant's reply is still
    streaming/not yet appended) when the caller speaks several short turns
    back-to-back, which is common since SpeechTimeoutUserTurnStopStrategy's
    short pause timeout ends turns eagerly. When that happened the buffer
    never reset and glued multiple already-answered turns' text into one
    query — confirmed live 2026-07-24 (call_917382700894_3d0ec1e2.log): four
    separate finished turns got concatenated into one query, which pulled in
    an unrelated chunk and caused the bot to proactively bring up a topic the
    caller never asked about in that turn.

    A later attempt reset on the raw VADUserStartedSpeakingFrame instead —
    that frame is emitted straight from VAD's own start_secs/stop_secs
    hangover timing, with no debounce against the higher-level logical turn.
    An ordinary mid-sentence pause longer than VAD_STOP_SECS (200ms) can
    flicker VAD stopped->started, re-firing that frame and wiping _turn_text
    before the rest of the same turn's fragments arrive — confirmed live
    2026-07-24 (call_917330771348_2f83e04b.log) still gluing separate
    finished turns together, so it didn't even fix the original bug.
    on_user_turn_started only fires once per logical turn (UserTurnController
    debounces on its own `_user_turn` flag), so that's the correct signal.

    When there are no hits, an explicit "none found" marker is injected
    instead of leaving no RAG system message at all — with no context and
    no explicit signal, the LLM would sometimes fall back to its own base
    persona on identity-style questions (e.g. answering about its own AI
    creators/founders instead of staying in character as the company rep).

    Turns that bot_processors.intent_router.classify_intent recognizes as
    clearly weather or mandi-price questions skip the search entirely —
    get_weather/get_price (bot_processors/weather_lookup.py, price_lookup.py)
    answer those directly, so there's no reason to also inject a KB passage
    that's at best redundant and at worst an unrelated chunk sitting in
    context next to the tool result.
    """

    def __init__(self, context: LLMContext, serializer=None, **kwargs):
        super().__init__(**kwargs)
        self._context = context
        # Logs each search as a fact_toolcalls row (see db/schema.sql) — this is
        # the only "tool" this bot has, there's no real LLM function-calling.
        # None call_id (before Smartflow's 'start' event) is a no-op in
        # log_tool_call, so this is safe from turn one.
        self._serializer = serializer
        self._pending_text = ""
        self._pending_task: "asyncio.Task | None" = None
        # Raw STT text accumulated across fragments of the current, still
        # unanswered user turn. Reset once the assistant has replied.
        self._turn_text = ""

    def reset_turn(self) -> None:
        if self._pending_task and not self._pending_task.done():
            self._pending_task.cancel()
        self._pending_task = None
        self._pending_text = ""
        self._turn_text = ""

    def _fire_speculative(self, text: str):
        if text == self._pending_text:
            return
        if self._pending_task and not self._pending_task.done():
            self._pending_task.cancel()
        self._pending_text = text
        speculative_intent = classify_intent(text)
        logger.debug(f"🧭 intent_router[speculative]: \"{text}\" -> {speculative_intent}")
        if speculative_intent in ("weather", "price"):
            self._pending_task = None
            return
        self._pending_task = asyncio.create_task(asyncio.to_thread(rag_search, text))

    async def _get_results(self, final_text: str):
        task = self._pending_task
        try:
            if task is not None:
                return await task
            return await asyncio.to_thread(rag_search, final_text)
        except Exception as e:
            logger.warning(f"⚠️ RAG search failed, continuing without context: {e}")
            return []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InterimTranscriptionFrame) and frame.text.strip():
            self._fire_speculative(frame.text.strip())

        elif isinstance(frame, TranscriptionFrame) and frame.text.strip():
            fragment_text = frame.text.strip()

            is_continuation = bool(self._turn_text)
            query_text = f"{self._turn_text} {fragment_text}".strip() if is_continuation else fragment_text
            self._turn_text = query_text

            intent = classify_intent(query_text)
            logger.info(f"🧭 intent_router[final]: \"{query_text}\" -> {intent}")
            _t0 = time.monotonic()
            if intent in ("weather", "price"):
                # get_weather/get_price (bot_processors/weather_lookup.py,
                # price_lookup.py) will answer this turn directly — skip the
                # search entirely rather than risk an incidental, unrelated
                # KB passage (or the "none found" marker, whose out-of-scope
                # framing doesn't apply here anyway) sitting in context
                # alongside the tool result.
                if self._pending_task and not self._pending_task.done():
                    self._pending_task.cancel()
                self._pending_task = None
                results = None
            elif is_continuation:
                # The speculative task (if any) only ever searched the latest
                # fragment in isolation, not the full accumulated turn text —
                # can't reuse it here, search fresh on the full turn text.
                self._pending_task = None
                results = await asyncio.to_thread(rag_search, query_text)
            else:
                results = await self._get_results(query_text)
            exec_ms = int((time.monotonic() - _t0) * 1000)
            self._pending_text = ""
            self._pending_task = None

            if self._serializer is not None and self._serializer.call_id:
                track_task(log_tool_call(
                    self._serializer.call_id,
                    "rag_search",
                    query_text,
                    f"skipped ({intent} intent)" if results is None
                    else (f"{len(results)} chunk(s) found" if results else "no chunks found"),
                    results is None or bool(results),
                    exec_ms,
                ))

            # Drop any RAG passages injected for a previous turn before adding
            # this turn's (or leaving none, if this turn had no hits, or this
            # turn's search was skipped for a weather/price intent).
            self._context._messages[:] = [
                m for m in self._context._messages
                if not (
                    m.get("role") == "system"
                    and isinstance(m.get("content"), str)
                    and m["content"].startswith(_RAG_MSG_PREFIX)
                )
            ]

            if results is None:
                logger.info(f"📚 RAG search skipped for \"{query_text}\" — classified as {intent} intent")
            elif results:
                passages = "\n\n".join(r["text"] for r in results)
                content = f"{_RAG_PASSAGE_PREFIX}{passages}"
                self._context.add_message({
                    "role": "system",
                    "content": content,
                })
                logger.info(f"📚 RAG injected {len(results)} chunk(s) for: \"{query_text}\"")
                logger.info(
                    "📤 Final context sent to LLM:\n"
                    + "\n".join(
                        f"  #{i} id={r['id']} rrf_score={r['score']:.5f} "
                        f"retriever={r.get('retriever', '')} text={r['text']!r}"
                        for i, r in enumerate(results, 1)
                    )
                )
            else:
                self._context.add_message({
                    "role": "system",
                    "content": _RAG_EMPTY_MARKER,
                })
                logger.info(f"📚 RAG found no chunks for: \"{query_text}\" — injected empty marker")

        await self.push_frame(frame, direction)
