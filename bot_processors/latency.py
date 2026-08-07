"""STT -> LLM -> TTS latency measurement and TTS cost tracking."""

import os
import time

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    MetricsFrame,
    StartFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import LLMUsageMetricsData, STTUsageMetricsData
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from bot_processors.call_db import backfill_llm_total_ms, log_performance_metric
from bot_processors.task_tracker import track_task


class _LatencyState:
    """Shared timing state between the two LatencyLogger instances."""
    __slots__ = (
        "t_stt", "t_llm", "t_tts", "t_interim_start", "interim_words", "call_active",
        "t_user_stopped", "stt_ms", "llm_total_ms", "detected_language", "turn_latencies_ms",
        "last_final_text", "last_final_ts", "last_final_consumed", "stt_usage_seconds",
        "llm_prompt_tokens", "llm_completion_tokens", "tts_audio_seconds",
    )
    def __init__(self):
        self.t_stt = 0.0
        self.t_llm = 0.0
        self.t_tts = 0.0
        self.t_interim_start = 0.0
        self.interim_words = 0
        self.call_active = False
        self.t_user_stopped = 0.0     # VAD "user stopped speaking" — start of this turn's STT
        self.stt_ms = None            # time from t_user_stopped to the final transcript
        self.llm_total_ms = None      # time from first token to full response done
        self.detected_language = ""   # most recent STT-detected language (e.g. "te")
        # One entry per completed turn (STT-final -> TTS-first-audio), same
        # total_response_ms already written to fact_performance — kept here
        # too so save_conversation_messages can attach it to the matching
        # assistant row in fact_conversations at call-end. See Bot.py's
        # _run_finalize.
        self.turn_latencies_ms = []
        # Independent copy of the last finalized STT transcript, kept outside
        # the LLMUserAggregator's own buffer. Confirmed live (call_919390427476_
        # 0ccb2fd9.log, 2026-08-05) that the aggregator can hit its forced
        # "on_user_turn_stop_timeout" fallback (strategy=None) with an EMPTY
        # aggregation even though this same TranscriptionFrame reached
        # LatencyLogger seconds earlier -- almost certainly the VAD-restart
        # mid-utterance case already documented in Bot.py wiping the
        # aggregator's `_full_user_turn_aggregation` before it's flushed.
        # Bot.py's on_user_turn_stopped handler uses this as a recovery copy
        # so a real answer isn't silently discarded as dead air.
        self.last_final_text = ""
        self.last_final_ts = 0.0
        self.last_final_consumed = True
        # Running total of client-measured STT audio-seconds this call, from
        # pipecat's built-in STTUsageMetricsData (added in pipecat-ai 1.7.0 —
        # see STTService.emit_stt_usage_metrics). Soniox reports this
        # incrementally per final transcript plus a trailing flush on
        # stop/cancel; summed here for the call-end cost summary, same as
        # CallCostTracker already does for TTS characters.
        self.stt_usage_seconds = 0.0
        # Running Gemini token totals this call, from LLMUsageMetricsData —
        # confirmed GoogleLLMService (which GoogleVertexLLMService/
        # _FastVertexLLMService extend) reports real prompt_tokens/
        # completion_tokens per turn, not an estimate.
        self.llm_prompt_tokens = 0
        self.llm_completion_tokens = 0
        # Running total of actual TTS OUTPUT audio duration this call —
        # computed from real PCM frame lengths (num_frames / sample_rate),
        # not estimated from text character count. Needed because Bakbak
        # bills per-minute of audio, not per-character (confirmed: 39 paise/
        # min) — a character-based estimate would be the wrong billing unit
        # entirely, not just a wrong number.
        self.tts_audio_seconds = 0.0


class LatencyLogger(FrameProcessor):
    """
    Measures STT → LLM → TTS latency across two pipeline positions.

    Place one instance right after STT (sees TranscriptionFrame /
    InterimTranscriptionFrame) and another right after TTS (sees
    TTSAudioRawFrame).  Both share the same _LatencyState object so
    timestamps recorded in the first instance are visible in the second.

    LLMTextFrame is handled by CallCostTracker (which sits between LLM and
    TTS and also has access to the shared state).
    """

    def __init__(self, state: _LatencyState, serializer=None, **kwargs):
        super().__init__(**kwargs)
        self._s = state
        # Only the "late" instance (placed after TTS) needs this — it's the
        # one that sees the TTS branch below, where a full turn's timings
        # (STT -> LLM first token -> TTS first audio) all become available
        # together. See Bot.py's `serializer.call_id`, populated once
        # Smartflow's 'start' event arrives.
        self._serializer = serializer

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if self._s.call_active:
            # ── User stopped speaking (VAD, early instance) — marks the start of
            # this turn's STT processing, so the next final transcript's delay
            # from here is the actual STT latency.
            if isinstance(frame, UserStoppedSpeakingFrame):
                self._s.t_user_stopped = time.monotonic()

            # ── STT interim: streaming word rate (early instance) ─────────────────
            elif isinstance(frame, InterimTranscriptionFrame) and frame.text.strip():
                now = time.monotonic()
                if not self._s.t_interim_start:
                    self._s.t_interim_start = now
                # STT providers send cumulative text in each interim frame (not just new
                # words), so always replace instead of accumulate for correct word count.
                self._s.interim_words = len(frame.text.split())
                elapsed = now - self._s.t_interim_start
                if elapsed > 0.5:
                    wps = self._s.interim_words / elapsed
                    logger.debug(
                        f"🎙️ STT stream | {wps:.1f} words/sec"
                        f"  ({self._s.interim_words} words in {elapsed:.1f}s)"
                    )

            # ── STT/LLM usage metrics (either instance — MetricsFrame is a
            # SystemFrame, not tied to pipeline position) ────────────────────
            elif isinstance(frame, MetricsFrame):
                for d in frame.data:
                    if isinstance(d, STTUsageMetricsData):
                        self._s.stt_usage_seconds += d.value.audio_seconds
                        logger.debug(
                            f"🎙️ STT usage | +{d.value.audio_seconds:.1f}s"
                            f"  → total {self._s.stt_usage_seconds:.1f}s this call"
                        )
                    elif isinstance(d, LLMUsageMetricsData):
                        self._s.llm_prompt_tokens += d.value.prompt_tokens
                        self._s.llm_completion_tokens += d.value.completion_tokens
                        logger.debug(
                            f"🧠 LLM usage | +{d.value.prompt_tokens} in / "
                            f"+{d.value.completion_tokens} out"
                            f"  → total {self._s.llm_prompt_tokens} in / "
                            f"{self._s.llm_completion_tokens} out this call"
                        )

            # ── STT final sentence (early instance) ──────────────────────────────
            elif isinstance(frame, TranscriptionFrame) and frame.text.strip():
                now = time.monotonic()
                self._s.t_stt = now
                self._s.t_llm = 0.0
                self._s.t_tts = 0.0
                self._s.llm_total_ms = None
                self._s.stt_ms = (now - self._s.t_user_stopped) * 1000 if self._s.t_user_stopped else None
                self._s.t_user_stopped = 0.0
                if frame.language:
                    self._s.detected_language = str(frame.language)
                self._s.last_final_text = frame.text
                self._s.last_final_ts = now
                self._s.last_final_consumed = False

                stream_dur = (now - self._s.t_interim_start) if self._s.t_interim_start else 0.0
                wps = (self._s.interim_words / stream_dur) if stream_dur > 0 else 0.0
                lang_tag = f"  [{frame.language}]" if frame.language else ""
                logger.info(
                    f"📝 STT  | sentence → \"{frame.text}\"{lang_tag}"
                    + (f"  stream {wps:.1f} w/s" if wps else "")
                )
                self._s.t_interim_start = 0.0
                self._s.interim_words = 0

            # ── TTS audio (late instance) — duration accumulated on EVERY chunk
            # (needed for per-minute cost tracking: Bakbak bills 39 paise/min of
            # audio, not per-character), first-chunk latency measured once per
            # turn same as before ──────────────────────────────────────────────
            elif isinstance(frame, TTSAudioRawFrame):
                chunk_seconds = frame.num_frames / frame.sample_rate if frame.sample_rate else 0.0
                self._s.tts_audio_seconds += chunk_seconds

                if self._s.t_stt and not self._s.t_tts:
                    self._s.t_tts = time.monotonic()
                    llm_gap = f"  (LLM→TTS {(self._s.t_tts - self._s.t_llm)*1000:.0f} ms)" if self._s.t_llm else ""
                    total_response_ms = (self._s.t_tts - self._s.t_stt) * 1000
                    self._s.turn_latencies_ms.append(int(total_response_ms))
                    logger.info(
                        f"🔊 TTS  | first audio {total_response_ms:.0f} ms after STT"
                        + llm_gap
                    )
                    if self._serializer is not None and self._serializer.call_id:
                        llm_ttft_ms = (self._s.t_llm - self._s.t_stt) * 1000 if self._s.t_llm else None
                        tts_ms = (self._s.t_tts - self._s.t_llm) * 1000 if self._s.t_llm else None
                        track_task(log_performance_metric(
                            self._serializer.call_id,
                            int(self._s.stt_ms) if self._s.stt_ms is not None else None,
                            int(llm_ttft_ms) if llm_ttft_ms is not None else None,
                            int(self._s.llm_total_ms) if self._s.llm_total_ms is not None else None,
                            int(tts_ms) if tts_ms is not None else None,
                            int(total_response_ms),
                        ))

        await self.push_frame(frame, direction)


# USD->INR display conversion — used ONLY for LLM cost (LLM_PRICE_PER_1M_*
# below), since that's the one price actually sourced in USD (Google AI
# Studio's published rate). STT_PRICE_PER_MIN and TTS_PRICE_PER_MIN are
# entered directly in rupees instead (see CallCostTracker) — Soniox and
# Bakbak's real rates are just as easy to set natively in ₹ as to convert,
# and Bakbak specifically bills in paise, not USD, so forcing a USD
# round-trip for it would be actively wrong, not just inconvenient.
# Deliberately no fallback numeric default — hardcoding "today's" rate would
# silently go stale (exchange rates drift daily). Unset (0) means "show
# plain $ for LLM cost, no conversion". Set USD_TO_INR_RATE in .env
# (look up the current rate yourself) to see ₹ there too.
USD_TO_INR_RATE = float(os.getenv("USD_TO_INR_RATE") or "0")


def _format_cost(usd: float) -> str:
    """LLM cost only — renders in ₹ if USD_TO_INR_RATE is configured, else
    falls back to $. STT/TTS use _format_native_cost instead (see there)."""
    if USD_TO_INR_RATE > 0:
        return f"₹{usd * USD_TO_INR_RATE:.4f}"
    return f"${usd:.4f}"


def _format_native_cost(rupees: float) -> str:
    """STT/TTS cost — the price knobs for these are already entered in ₹
    directly (see STT_PRICE_PER_MIN/TTS_PRICE_PER_MIN), so this just
    formats, no conversion involved."""
    return f"₹{rupees:.4f}"


class CallCostTracker(FrameProcessor):
    """Tracks per-call cost across all three paid services (STT, TTS, LLM)
    and logs a cost/min or cost-total figure for each at call-end.

    Also records LLM first-token time into the shared LatencyState (because
    LLMTextFrame is consumed by the TTS service and never reaches the late
    LatencyLogger instance).
    """

    # STT ₹/minute, entered natively in rupees — Soniox's real rate ($0.002/
    # min, from soniox.com/pricing for the stt-rt-v5 model this project
    # uses) converted at the USD_TO_INR_RATE checked 2026-08-07 (95.28) =
    # ₹0.1906/min. Unlike LLM below, this is NOT re-converted at display
    # time — it's just entered directly as a rupee figure, since that's what
    # you actually adjust day to day. Re-derive from Soniox's USD rate only
    # if you want to track their pricing changes; otherwise just edit this
    # number directly in .env when your own rate changes.
    STT_PRICE_PER_MIN = float(os.getenv("STT_PRICE_PER_MIN") or "0")
    # TTS ₹/minute of actual output audio (see LatencyLogger's
    # tts_audio_seconds — real PCM duration, not a character-count guess).
    # Bakbak bills per-minute of audio, not per-character, so this is the
    # right billing unit for this provider unlike the old char-based
    # DEFAULT_PRICE_PER_1K below. Confirmed rate: 39 paise/min = ₹0.39/min.
    TTS_PRICE_PER_MIN = float(os.getenv("TTS_PRICE_PER_MIN") or "0")
    # Legacy character-based pricing — kept only for providers that
    # genuinely bill per-character (e.g. Cartesia, if ever switched back
    # to). NOT used for cost calculation while TTS_PRICE_PER_MIN is set;
    # character count is still logged for visibility either way.
    DEFAULT_PRICE_PER_1K = float(
        os.getenv("TTS_PRICE_PER_1K_CHARS")
        or os.getenv("CARTESIA_PRICE_PER_1K_CHARS")
        or "0.065"
    )
    # Gemini $/million tokens — input and output are priced differently, so
    # two separate knobs. These stay USD (Google's own published rate) and
    # get converted via USD_TO_INR_RATE at display time, unlike STT/TTS
    # above — see USD_TO_INR_RATE's comment for why LLM is the one exception.
    LLM_PRICE_PER_1M_INPUT_TOKENS = float(os.getenv("LLM_PRICE_PER_1M_INPUT_TOKENS") or "0")
    LLM_PRICE_PER_1M_OUTPUT_TOKENS = float(os.getenv("LLM_PRICE_PER_1M_OUTPUT_TOKENS") or "0")

    def __init__(
        self, state: _LatencyState, echo_buffer: "EchoBuffer | None" = None,
        serializer=None, **kwargs,
    ):
        super().__init__(**kwargs)
        self._s = state
        self._echo_buf = echo_buffer
        self._serializer = serializer
        self._call_start: float = 0.0
        self._total_chars: int = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            self._call_start = time.monotonic()
            self._total_chars = 0

        elif isinstance(frame, LLMFullResponseEndFrame) and self._s.t_llm:
            self._s.llm_total_ms = (time.monotonic() - self._s.t_llm) * 1000
            # Multi-sentence responses: TTS first-audio (and its fact_performance
            # row) can fire before this, leaving llm_total_ms NULL there. Patch
            # it in now; a no-op if that row already had it or doesn't exist yet.
            if self._serializer is not None and self._serializer.call_id:
                track_task(backfill_llm_total_ms(
                    self._serializer.call_id, int(self._s.llm_total_ms),
                ))

        elif isinstance(frame, LLMTextFrame) and frame.text:
            # Feed LLM token to echo buffer so EchoFilter can catch the phone echo
            if self._echo_buf:
                self._echo_buf.add(frame.text)
            # ── LLM first token latency ──────────────────────────────────────
            if self._s.t_stt and not self._s.t_llm:
                self._s.t_llm = time.monotonic()
                logger.info(
                    f"⚡ LLM  | first token {(self._s.t_llm - self._s.t_stt)*1000:.0f} ms after STT"
                )
            # ── Character count (informational only) ──────────────────────────
            # No live $ figure here anymore: TTS cost is minute-based now (see
            # TTS_PRICE_PER_MIN), and audio duration isn't known until the
            # TTSAudioRawFrame chunks actually arrive — character count at
            # text-arrival time was the right moment to estimate cost under
            # the old per-character model, but not this one. Real-time running
            # TTS cost isn't shown live for this reason; the accurate total is
            # in log_summary() at call-end once all audio has been produced.
            chars = len(frame.text)
            self._total_chars += chars
            logger.debug(f"📝 TTS text | +{chars} chars → total {self._total_chars} chars")

        await self.push_frame(frame, direction)

    def reset(self):
        self._call_start = time.monotonic()
        self._total_chars = 0

    def log_summary(self):
        if not self._call_start:
            return
        elapsed_min = (time.monotonic() - self._call_start) / 60.0
        tts_seconds = self._s.tts_audio_seconds
        tts_label = os.getenv("TTS_PROVIDER", "TTS").upper()

        if self.TTS_PRICE_PER_MIN > 0:
            # Real per-minute billing (Bakbak) — cost from actual audio
            # duration, not character count.
            tts_cost = (tts_seconds / 60.0) * self.TTS_PRICE_PER_MIN
            rate = (tts_cost / elapsed_min) if elapsed_min > 0 else 0.0
            logger.info(
                f"💰 CALL END | {tts_label}: {tts_seconds:.1f}s audio"
                f" ({self._total_chars} chars)"
                f"  {_format_native_cost(tts_cost)} total  call {elapsed_min:.1f} min"
                f"  ~{_format_native_cost(rate)}/min"
            )
        elif self.DEFAULT_PRICE_PER_1K > 0 and self._total_chars > 0:
            # Legacy fallback: genuinely character-billed providers (e.g.
            # Cartesia) — only used when TTS_PRICE_PER_MIN isn't set.
            price_per_char = self.DEFAULT_PRICE_PER_1K / 1000.0
            total_cost = self._total_chars * price_per_char
            rate = (total_cost / elapsed_min) if elapsed_min > 0 else 0.0
            logger.info(
                f"💰 CALL END | {tts_label}: {self._total_chars} chars"
                f"  {_format_cost(total_cost)} total  call {elapsed_min:.1f} min"
                f"  ~{_format_cost(rate)}/min"
            )
        else:
            logger.info(
                f"💰 CALL END | {tts_label}: {tts_seconds:.1f}s audio"
                f" ({self._total_chars} chars)  call {elapsed_min:.1f} min"
            )
        # STT usage (pipecat-ai>=1.7.0's built-in STTUsageMetricsData) —
        # separate log line since it's a different provider/pipeline stage;
        # only shows a cost figure if STT_PRICE_PER_MIN is actually configured.
        stt_seconds = self._s.stt_usage_seconds
        if stt_seconds > 0:
            stt_label = os.getenv("STT_PROVIDER", "STT").upper()
            cost_str = ""
            if self.STT_PRICE_PER_MIN > 0:
                stt_cost = (stt_seconds / 60.0) * self.STT_PRICE_PER_MIN
                cost_str = f"  {_format_native_cost(stt_cost)} total"
            logger.info(
                f"🎙️ CALL END | {stt_label}: {stt_seconds:.1f}s audio submitted{cost_str}"
            )
        # LLM usage (pipecat-ai>=1.7.0's built-in LLMUsageMetricsData, real
        # token counts from Gemini — not an estimate). Separate log line,
        # same reasoning as STT above: only shows a cost figure if both
        # LLM_PRICE_PER_1M_INPUT_TOKENS/LLM_PRICE_PER_1M_OUTPUT_TOKENS are
        # configured; a price of 0 for one just contributes $0 for that side
        # rather than hiding the whole figure, so setting only one still
        # gives a (partial) cost rather than nothing.
        prompt_tok, completion_tok = self._s.llm_prompt_tokens, self._s.llm_completion_tokens
        if prompt_tok or completion_tok:
            llm_label = os.getenv("VERTEX_MODEL", "LLM")
            cost_str = ""
            if self.LLM_PRICE_PER_1M_INPUT_TOKENS > 0 or self.LLM_PRICE_PER_1M_OUTPUT_TOKENS > 0:
                llm_cost = (
                    (prompt_tok / 1_000_000) * self.LLM_PRICE_PER_1M_INPUT_TOKENS
                    + (completion_tok / 1_000_000) * self.LLM_PRICE_PER_1M_OUTPUT_TOKENS
                )
                cost_str = f"  {_format_cost(llm_cost)} total"
            logger.info(
                f"🧠 CALL END | {llm_label}: {prompt_tok} in / {completion_tok} out tokens{cost_str}"
            )
