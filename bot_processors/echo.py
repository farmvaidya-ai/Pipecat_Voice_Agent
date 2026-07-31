"""Phone-line echo detection/filtering and STT transcript correction."""

import re
import time
import unicodedata

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from providers.transcript_correction import correct_transcript


class EchoBuffer:
    """
    Stores recent bot speech text so EchoFilter can detect phone echo.

    Phone echo path:
      Bot TTS audio → phone speaker → phone mic → STT → TranscriptionFrame
    This buffer lets EchoFilter drop those frames before they reach the LLM.

    TTL: 8 seconds covers the slowest PSTN echo delay seen in testing.
    """

    def __init__(self, ttl_seconds: float = 8.0):
        self._ttl = ttl_seconds
        self._entries: list[tuple[float, str]] = []  # (timestamp, normalized_text)

    def add(self, text: str) -> None:
        """Store a piece of bot speech (call for every token/phrase)."""
        normalized = _echo_normalize(text)
        if normalized:
            self._entries.append((time.monotonic(), normalized))

    def clear(self) -> None:
        """Drop all stored entries (call between calls on a reused pipeline)."""
        self._entries.clear()

    def is_echo(self, transcript: str, threshold: float = 0.70) -> bool:
        """
        Return True if ≥ threshold fraction of the transcript's words
        appear in recent bot speech.

        threshold=0.70 means 70% word overlap → marked as echo.
        Keeps false positives low while catching most real echoes.
        """
        now = time.monotonic()
        self._entries = [(t, txt) for t, txt in self._entries if now - t < self._ttl]

        if not self._entries:
            return False

        norm = _echo_normalize(transcript)
        trans_words = set(norm.split())
        if not trans_words:
            return False

        all_bot_words: set[str] = set()
        for _, bot_text in self._entries:
            all_bot_words.update(bot_text.split())

        overlap = len(trans_words & all_bot_words)
        return (overlap / len(trans_words)) >= threshold

    def reset(self) -> None:
        self._entries.clear()


def _echo_normalize(text: str) -> str:
    """
    Lowercase + strip punctuation (unicode-safe) for word-level matching.

    Deliberately strips by Unicode category (P = punctuation) instead of
    `\\w`, since `\\w` excludes category M (combining marks) — Telugu and
    other Indic abugidas build syllables out of base consonants + vowel-sign
    marks, so a `\\w`-based strip deletes every vowel sign and virama and
    shatters real words into bare consonant fragments. Those fragments
    collide across totally unrelated sentences (~36 Telugu consonants total)
    and were causing false-positive echo drops.
    """
    text = text.lower()
    text = ''.join(' ' if unicodedata.category(ch).startswith('P') else ch for ch in text)
    return re.sub(r'\s+', ' ', text).strip()


class TranscriptCorrector(FrameProcessor):
    """
    Sits right after STT, before everything else (echo filter, RAG, LLM).

    Rewrites TranscriptionFrame/InterimTranscriptionFrame text using the
    domain-term dictionary (fuzzy + phonetic match — see
    providers/transcript_correction.py) to catch misspellings/mispronunciations
    that Soniox's native context biasing doesn't fully resolve. One fix point
    here means both RAG retrieval and the LLM see the corrected text.
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame)) and frame.text.strip():
            corrected = correct_transcript(frame.text)
            if corrected != frame.text:
                logger.info(f"📝 DICT CORRECTION | \"{frame.text}\" -> \"{corrected}\"")
                frame.text = corrected

        await self.push_frame(frame, direction)


class EchoFilter(FrameProcessor):
    """
    Sits between STT and context_aggregator.user().

    Two jobs:
    1. Watches TTSSpeakFrame flowing downstream (greeting) → adds text to EchoBuffer.
    2. Checks every TranscriptionFrame against EchoBuffer → drops if it looks like echo.

    LLMTextFrame tokens (bot responses) are fed into the same buffer by CallCostTracker
    because those frames appear AFTER this processor in the pipeline.
    """

    def __init__(self, echo_buffer: EchoBuffer, **kwargs):
        super().__init__(**kwargs)
        self._buf = echo_buffer

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TTSSpeakFrame) and frame.text:
            # Greeting flows through here as TTSSpeakFrame — store it so echo can be caught
            self._buf.add(frame.text)

        elif isinstance(frame, TranscriptionFrame) and frame.text.strip():
            if self._buf.is_echo(frame.text):
                logger.warning(f"🔇 ECHO DROPPED | \"{frame.text}\"")
                return  # Drop the frame — LLM never sees it

        await self.push_frame(frame, direction)
