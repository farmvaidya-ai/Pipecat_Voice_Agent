"""
Custom Bakbak (Raya) TTS service for pipecat.

API: POST https://hub.getraya.app/v1/text-to-speech
Auth: X-API-Key header
Body: { text, voice_id, model, language }

Bakbak returns ADTS AAC audio (0xFF 0xF1 sync word).
PyAV decodes it to raw PCM s16le at the pipeline sample rate before
the frames are yielded into pipecat.

Bakbak also enforces a per-request character limit (~60 code points is safe
for Indian scripts where combining marks inflate Python len()).  Long text is
split at sentence / clause / word boundaries and each chunk is a separate call.
"""

import asyncio
import io
import re
from collections.abc import AsyncGenerator
from dataclasses import dataclass

import aiohttp
import av
from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language

from providers.word_batch_aggregator import WordBatchTextAggregator


_LANG_TO_BAKBAK: dict[Language, str] = {
    Language.EN: "en",
    Language.HI: "hi",
    Language.TE: "te",
    Language.TA: "ta",
    Language.KN: "kn",
    Language.ML: "ml",
    Language.BN: "bn",
    Language.MR: "mr",
    Language.GU: "gu",
    Language.PA: "pa",
}

_DEFAULT_MAX_CHARS = 60  # safe for Telugu/Hindi (60 × 3 UTF-8 bytes = 180 bytes)

# Per-request timeout and retry budget for the Bakbak HTTP call. Without an
# explicit timeout, aiohttp defaults to 300s — a network blip would otherwise
# stall the call for up to 5 minutes before erroring.
_REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=12)
_MAX_RETRIES = 3

# Module-level (not instance-level) cache of pre-synthesized PCM audio for
# static, repeated-every-call text (e.g. the fixed greeting). Every real call
# builds a brand-new BakbakTTSService instance, so this must live outside the
# instance to survive from the boot-time warmup call to every later call's
# run_tts(). Populated once via warm_static_text() before any real caller
# connects; never mutated afterward, so concurrent reads in run_tts() are safe.
# Keyed by every payload field that changes Bakbak's output.
_static_audio_cache: dict[tuple, list[bytes]] = {}


@dataclass
class BakbakTTSSettings(TTSSettings):
    """Settings for BakbakTTSService."""
    pass


class BakbakTTSService(TTSService):
    """HTTP TTS client for Bakbak / Raya (https://hub.getraya.app).

    Bakbak returns ADTS AAC audio; this service decodes it to PCM via PyAV
    before yielding TTSAudioRawFrame so the pipeline receives correct audio.
    """

    Settings = BakbakTTSSettings
    _settings: BakbakTTSSettings

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str = "d6a002d0-230c-49b1-a137-b8a7d564b1ae",  # Raya
        language: str = "en",
        model: str = "standard",
        codec: str = "pcm",
        base_url: str = "https://hub.getraya.app/v1/text-to-speech",
        max_chars: int = _DEFAULT_MAX_CHARS,
        # When set, text is flushed to TTS every `word_batch_size` words
        # (extended to a nearby clause boundary) instead of waiting for a
        # full sentence — cuts time-to-first-audio at the cost of slightly
        # choppier prosody at chunk boundaries. None = default sentence
        # aggregation (wait for full sentence, as before).
        word_batch_size: int | None = None,
        clause_lookahead_words: int = 3,
        sample_rate: int | None = None,
        settings: BakbakTTSSettings | None = None,
        **kwargs,
    ):
        default_settings = self.Settings(
            model=model,
            voice=voice_id,
            language=language,
        )
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=sample_rate,
            push_start_frame=True,
            push_stop_frames=True,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._base_url = base_url
        self._codec = codec
        self._max_chars = max_chars
        self._session: aiohttp.ClientSession | None = None

        if word_batch_size is not None:
            self._text_aggregator = WordBatchTextAggregator(
                word_batch_size=word_batch_size,
                clause_lookahead_words=clause_lookahead_words,
            )

    def can_generate_metrics(self) -> bool:
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        return _LANG_TO_BAKBAK.get(language, "en")

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def start(self, frame):
        await super().start(frame)
        if not self._session or self._session.closed:
            self._session = aiohttp.ClientSession()

    async def stop(self, frame):
        await super().stop(frame)
        await self._close_session()

    async def cancel(self, frame):
        await super().cancel(frame)
        await self._close_session()

    async def _close_session(self):
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None

    # ── AAC → PCM decoding ───────────────────────────────────────────────────

    def _decode_aac(self, aac_data: bytes) -> bytes:
        """Decode ADTS AAC bytes → raw PCM s16le at self.sample_rate, mono."""
        container = av.open(io.BytesIO(aac_data))
        try:
            audio_stream = next(s for s in container.streams if s.type == "audio")
            resampler = av.AudioResampler(
                format="s16",
                layout="mono",
                rate=self.sample_rate,
            )
            pcm_parts: list[bytes] = []
            for frame in container.decode(audio_stream):
                for resampled in resampler.resample(frame):
                    pcm_parts.append(bytes(resampled.planes[0]))
            for resampled in resampler.resample(None):
                pcm_parts.append(bytes(resampled.planes[0]))
            return b"".join(pcm_parts)
        finally:
            container.close()

    # ── text splitting ────────────────────────────────────────────────────────

    def _split_text(self, text: str) -> list[str]:
        """Split text into chunks ≤ max_chars at natural boundaries."""
        if len(text) <= self._max_chars:
            return [text]

        chunks: list[str] = []
        sentences = re.split(r'(?<=[।.!?])\s+', text.strip())

        current = ""
        for sentence in sentences:
            if not sentence:
                continue
            if len(sentence) > self._max_chars:
                clauses = re.split(r'(?<=[,;])\s+', sentence)
                for clause in clauses:
                    if not clause:
                        continue
                    if len(clause) > self._max_chars:
                        for word in clause.split():
                            candidate = (current + " " + word).strip() if current else word
                            if len(candidate) > self._max_chars and current:
                                chunks.append(current)
                                current = word
                            else:
                                current = candidate
                    else:
                        candidate = (current + " " + clause).strip() if current else clause
                        if len(candidate) > self._max_chars and current:
                            chunks.append(current)
                            current = clause
                        else:
                            current = candidate
            else:
                candidate = (current + " " + sentence).strip() if current else sentence
                if len(candidate) > self._max_chars and current:
                    chunks.append(current)
                    current = sentence
                else:
                    current = candidate

        if current:
            chunks.append(current)
        return [c for c in chunks if c.strip()]

    # ── HTTP call with retry ─────────────────────────────────────────────────

    async def _call_bakbak(self, payload: dict, chunk: str) -> bytes | None:
        """POST one chunk to Bakbak, retrying on transient errors.

        Returns raw response bytes on success, or None on permanent failure
        (an ErrorFrame-worthy message is logged; caller decides how to signal
        it downstream). 4xx errors are deterministic and not retried; 5xx and
        connection-level errors are retried with a short backoff, recreating
        the session first since the pooled connection may be stale.
        """
        headers = {
            "X-API-Key": self._api_key,
            "Content-Type": "application/json",
        }
        last_error = "unknown error"

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                if not self._session or self._session.closed:
                    self._session = aiohttp.ClientSession()

                async with self._session.post(
                    self._base_url, json=payload, headers=headers, timeout=_REQUEST_TIMEOUT
                ) as resp:
                    content_type = resp.headers.get("Content-Type", "")

                    if resp.status != 200:
                        error_text = await resp.text(errors="ignore")
                        last_error = f"HTTP {resp.status}: {error_text[:200]}"
                        logger.warning(f"Bakbak TTS attempt {attempt}/{_MAX_RETRIES} → {last_error}")
                        if 400 <= resp.status < 500:
                            logger.error(f"Bakbak TTS error (no retry): {last_error}")
                            return None
                        if attempt < _MAX_RETRIES:
                            await asyncio.sleep(0.4 * attempt)
                        continue

                    audio_data = await resp.read()

                    # A 200 with a tiny/JSON body is Bakbak reporting an error
                    # without a proper error status — treat it as a failure.
                    if not audio_data or len(audio_data) < 200 or "json" in content_type.lower():
                        decoded = audio_data.decode("utf-8", errors="replace")
                        last_error = f"non-audio response (ct={content_type!r}, {len(audio_data)}B): {decoded[:300]}"
                        logger.error(f"Bakbak TTS returned {last_error}")
                        return None

                    logger.debug(f"Bakbak TTS: {len(audio_data)}B for [{chunk[:25]}] (attempt {attempt})")
                    return audio_data

            except (aiohttp.ClientConnectionError, aiohttp.ServerDisconnectedError, asyncio.TimeoutError) as e:
                last_error = f"{type(e).__name__}: {e}"
                logger.warning(f"Bakbak TTS connection error (attempt {attempt}/{_MAX_RETRIES}): {e}")
                if self._session and not self._session.closed:
                    await self._session.close()
                self._session = aiohttp.ClientSession()
                if attempt < _MAX_RETRIES:
                    await asyncio.sleep(0.3 * attempt)

        logger.error(f"Bakbak TTS failed after {_MAX_RETRIES} attempts: {last_error}")
        return None

    # ── static-text cache ────────────────────────────────────────────────────

    def _cache_key(self, chunk: str) -> tuple:
        return (
            chunk,
            self._settings.voice,
            self._settings.model,
            self._settings.language,
            self._codec,
            self.sample_rate,
        )

    async def _fetch_chunk_pcm(self, chunk: str) -> bytes | None:
        """POST one chunk to Bakbak and return decoded PCM, or None on failure."""
        payload = {
            "text": chunk,
            "voice_id": self._settings.voice,
            "model": self._settings.model,
            "language": self._settings.language,
            "codec": self._codec,
            "sample_rate": self.sample_rate,
        }
        audio_data = await self._call_bakbak(payload, chunk)
        if audio_data is None:
            return None

        if self._codec == "pcm":
            return audio_data

        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(None, self._decode_aac, audio_data)
        except Exception as exc:
            logger.error(f"Bakbak TTS AAC decode error while warming static text: {exc}")
            return None

    async def warm_static_text(self, text: str) -> bool:
        """Pre-synthesize `text` once at boot and cache the PCM so every real
        call's run_tts() can skip Bakbak's network call for it entirely.

        Caller must set self._sample_rate to the value real calls will use
        (StartFrame hasn't happened yet on a throwaway warmup instance, so
        self.sample_rate would otherwise still be 0) before calling this.
        """
        chunks = self._split_text(text)
        pcm_chunks: list[bytes] = []
        for chunk in chunks:
            pcm = await self._fetch_chunk_pcm(chunk)
            if pcm is None:
                logger.warning(f"Bakbak: failed to warm static text chunk [{chunk[:40]}]")
                return False
            pcm_chunks.append(pcm)
            _static_audio_cache[self._cache_key(chunk)] = [pcm]
        logger.info(f"🔥 Bakbak greeting audio cached ({len(chunks)} chunk(s), {sum(len(p) for p in pcm_chunks)}B)")
        return True

    # ── TTS ───────────────────────────────────────────────────────────────────

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        chunks = self._split_text(text)
        logger.debug(f"{self}: Generating TTS [{text}] → {len(chunks)} chunk(s)")

        measuring_ttfb = True

        for chunk in chunks:
            try:
                cached = _static_audio_cache.get(self._cache_key(chunk))
                if cached is not None:
                    if measuring_ttfb:
                        await self.stop_ttfb_metrics()
                        measuring_ttfb = False
                    pcm_data = cached[0]
                    for i in range(0, len(pcm_data), self.chunk_size):
                        yield TTSAudioRawFrame(
                            pcm_data[i : i + self.chunk_size],
                            self.sample_rate,
                            1,
                            context_id=context_id,
                        )
                    continue

                await self.start_tts_usage_metrics(chunk)

                payload = {
                    "text": chunk,
                    "voice_id": self._settings.voice,
                    "model": self._settings.model,
                    "language": self._settings.language,
                    "codec": self._codec,
                    "sample_rate": self.sample_rate,
                }
                audio_data = await self._call_bakbak(payload, chunk)
                if audio_data is None:
                    yield ErrorFrame(error=f"Bakbak TTS failed for chunk [{chunk[:40]}]")
                    return

                if self._codec == "pcm":
                    # Bakbak sends raw PCM at the requested sample_rate —
                    # ready to yield directly, no decode step needed.
                    pcm_data = audio_data
                else:
                    # Decode AAC → PCM off the event loop so it doesn't block
                    loop = asyncio.get_event_loop()
                    try:
                        pcm_data = await loop.run_in_executor(None, self._decode_aac, audio_data)
                    except Exception as exc:
                        yield ErrorFrame(error=f"Bakbak TTS AAC decode error: {exc}")
                        return

                if measuring_ttfb:
                    await self.stop_ttfb_metrics()
                    measuring_ttfb = False

                # Yield PCM in pipeline-sized chunks
                for i in range(0, len(pcm_data), self.chunk_size):
                    yield TTSAudioRawFrame(
                        pcm_data[i : i + self.chunk_size],
                        self.sample_rate,
                        1,
                        context_id=context_id,
                    )

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                yield ErrorFrame(error=f"Bakbak TTS unexpected error: {exc}")
                return
