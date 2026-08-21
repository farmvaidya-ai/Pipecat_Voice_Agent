"""
Custom Svara (Kenpath Labs) TTS service for pipecat.

Svara ships its own pipecat wrapper (`svara.pipecat.SvaraTTSService`), but it
targets an older pipecat-ai `TTSService` contract: its `run_tts(self, text)`
has no `context_id` parameter, while pipecat-ai 1.7.0 (installed here) calls
`run_tts(self, text, context_id)` — the per-turn audio-context mechanism
every other TTS provider in this project relies on. Using the bundled
wrapper as-is raises a TypeError the moment the pipeline calls run_tts (its
own docstring flags it as "Beta — validate against your pinned pipecat-ai
version", so this isn't a surprise, just unverified against this repo's
version). Confirmed by reading both source trees directly rather than
guessing — see pipecat/services/tts_service.py's run_tts signature vs.
svara/pipecat/__init__.py in the svara-voice package.

So: this adapter uses the SDK's underlying `AsyncSvara` HTTP client directly
(streaming, retry/backoff on 429s and 5xx, auth — all handled there) wrapped
in the same context_id-aware contract this project's other custom TTS
adapter (providers/bakbak_tts.py) already uses successfully in production.

API: POST https://api.kenpathlabs.com/v1/audio/speech
Auth: `xi-api-key` header (set internally by AsyncSvara from SVARA_API_KEY /
the api_key passed in — the docs page also advertises `Authorization:
Bearer`, but xi-api-key is what the SDK actually sends).

Svara streams headerless PCM s16le mono directly at the requested
sample_rate when response_format="pcm" — unlike Bakbak (AAC), there's no
decode step needed here.
"""

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass

from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService

from svara._client import AsyncSvara
from svara.exceptions import SvaraError


@dataclass
class SvaraTTSSettings(TTSSettings):
    """Settings for SvaraTTSService (voice/model/language inherited from TTSSettings)."""
    pass


class SvaraTTSService(TTSService):
    """HTTP TTS client for Svara (Kenpath Labs, https://kenpathlabs.com).

    Wraps the official `svara-voice` SDK's AsyncSvara client directly instead
    of the SDK's own `svara.pipecat.SvaraTTSService` — see module docstring
    for why. Svara streams raw PCM directly, so frames are pushed as bytes
    arrive with no decode step.
    """

    Settings = SvaraTTSSettings
    _settings: SvaraTTSSettings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        voice: str,
        language: str | None = None,
        model: str = "svara-1",
        response_format: str = "pcm",
        speed: float | None = None,
        sample_rate: int | None = None,
        settings: SvaraTTSSettings | None = None,
        **kwargs,
    ):
        default_settings = self.Settings(
            model=model,
            voice=voice,
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

        # response_format/speed aren't part of pipecat's TTSSettings schema
        # (voice/model/language only), so they're kept as plain instance
        # attributes rather than shoehorned into _settings.
        self._response_format = response_format
        self._speed = speed
        self._client = AsyncSvara(api_key=api_key, base_url=base_url)

    def can_generate_metrics(self) -> bool:
        return True

    async def stop(self, frame):
        await super().stop(frame)
        await self._client.aclose()

    async def cancel(self, frame):
        await super().cancel(frame)
        await self._client.aclose()

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        logger.debug(f"{self}: Generating TTS [{text}]")

        await self.start_tts_usage_metrics(text)
        measuring_ttfb = True

        try:
            async for chunk in self._client.speech.stream(
                input=text,
                voice=self._settings.voice,
                model=self._settings.model,
                response_format=self._response_format,
                # self.sample_rate is 0 until StartFrame has run; None lets
                # the server use its own default (24kHz) in that edge case
                # instead of requesting an invalid 0Hz stream.
                sample_rate=self.sample_rate or None,
                speed=self._speed,
                language=self._settings.language,
                # Sized to match pipecat's own recommended chunk size (0.5s
                # of audio, always an even byte count) so every chunk but
                # possibly the last stays aligned to 2-byte PCM samples —
                # same approach OpenAITTSService uses for its own streaming.
                chunk_size=self.chunk_size,
            ):
                if measuring_ttfb:
                    await self.stop_ttfb_metrics()
                    measuring_ttfb = False
                yield TTSAudioRawFrame(chunk, self.sample_rate, 1, context_id=context_id)
        except SvaraError as e:
            logger.error(f"Svara TTS error: {e}")
            yield ErrorFrame(error=f"Svara TTS error: {e}")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(f"Svara TTS unexpected error: {exc}")
            yield ErrorFrame(error=f"Svara TTS unexpected error: {exc}")
