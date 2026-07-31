"""STT -> LLM -> TTS latency measurement and TTS cost tracking."""

import os
import time

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    StartFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from bot_processors.call_db import backfill_llm_total_ms, log_performance_metric
from bot_processors.task_tracker import track_task


class _LatencyState:
    """Shared timing state between the two LatencyLogger instances."""
    __slots__ = (
        "t_stt", "t_llm", "t_tts", "t_interim_start", "interim_words", "call_active",
        "t_user_stopped", "stt_ms", "llm_total_ms", "detected_language", "turn_latencies_ms",
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

                stream_dur = (now - self._s.t_interim_start) if self._s.t_interim_start else 0.0
                wps = (self._s.interim_words / stream_dur) if stream_dur > 0 else 0.0
                lang_tag = f"  [{frame.language}]" if frame.language else ""
                logger.info(
                    f"📝 STT  | sentence → \"{frame.text}\"{lang_tag}"
                    + (f"  stream {wps:.1f} w/s" if wps else "")
                )
                self._s.t_interim_start = 0.0
                self._s.interim_words = 0

            # ── TTS first audio chunk (late instance) ────────────────────────────
            elif isinstance(frame, TTSAudioRawFrame) and self._s.t_stt and not self._s.t_tts:
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


class CallCostTracker(FrameProcessor):
    """Tracks TTS character spend and logs $/min.

    Also records LLM first-token time into the shared LatencyState (because
    LLMTextFrame is consumed by the TTS service and never reaches the late
    LatencyLogger instance).
    """

    # Checks TTS_PRICE_PER_1K_CHARS first for provider-agnostic config;
    # falls back to CARTESIA_PRICE_PER_1K_CHARS for backward compatibility.
    DEFAULT_PRICE_PER_1K = float(
        os.getenv("TTS_PRICE_PER_1K_CHARS")
        or os.getenv("CARTESIA_PRICE_PER_1K_CHARS")
        or "0.065"
    )

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
        self._price_per_char = self.DEFAULT_PRICE_PER_1K / 1000.0

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
            # ── Cost tracking ────────────────────────────────────────────────
            chars = len(frame.text)
            self._total_chars += chars
            cost = self._total_chars * self._price_per_char
            elapsed_min = (time.monotonic() - self._call_start) / 60.0
            rate = (cost / elapsed_min) if elapsed_min > 0 else 0.0
            logger.debug(
                f"💰 TTS cost | +{chars} chars → total {self._total_chars} chars"
                f"  ${cost:.5f} total  (${rate:.4f}/min)"
            )

        await self.push_frame(frame, direction)

    def reset(self):
        self._call_start = time.monotonic()
        self._total_chars = 0

    def log_summary(self):
        if not self._call_start:
            return
        elapsed_min = (time.monotonic() - self._call_start) / 60.0
        total_cost = self._total_chars * self._price_per_char
        rate = (total_cost / elapsed_min) if elapsed_min > 0 else 0.0
        tts_label = os.getenv("TTS_PROVIDER", "TTS").upper()
        logger.info(
            f"💰 CALL END | {tts_label}: {self._total_chars} chars"
            f"  ${total_cost:.4f} total  call {elapsed_min:.1f} min"
            f"  ~${rate:.4f}/min"
        )
