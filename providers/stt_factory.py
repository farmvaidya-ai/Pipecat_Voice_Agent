"""
STT Factory — reads STT_PROVIDER from .env and returns the configured service.

Supported providers (set STT_PROVIDER= in .env):
    soniox   → SonioxSTTService   (default)
    sarvam   → SarvamSTTService
    gnani    → GnaniSTTService    (Gnani.ai Prisma v2.5)

Adding a new provider:
    1. Add its env vars to .env (e.g. NEWPROVIDER_STT_API_KEY=, NEWPROVIDER_STT_MODEL=)
    2. Add an elif branch in create_stt() that calls a new _create_newprovider_stt() helper
    3. Write the helper function below following the same pattern
    4. Update the error message in create_stt() to list the new provider
"""

import asyncio
import os
from typing import Optional, List

from loguru import logger
from pipecat.transcriptions.language import Language


def create_stt(
    language_hints: Optional[List[Language]] = None,
    language_hints_strict: bool = False,
    enable_language_identification: bool = True,
):
    """
    Instantiate and return the STT service selected by STT_PROVIDER in .env.

    Args:
        language_hints:                 List of Language enums to guide detection.
        language_hints_strict:          If True, only transcribe hinted languages.
        enable_language_identification: Attach detected language to every frame.

    Returns:
        Configured STT service instance ready for pipeline use.

    Raises:
        ValueError:  Required env var is missing.
        ImportError: Provider package is not installed.
        ValueError:  STT_PROVIDER names an unsupported provider.
    """
    provider = os.getenv("STT_PROVIDER", "soniox").lower().strip()
    logger.info(f"[STT Factory] provider={provider.upper()}")

    if provider == "soniox":
        return _create_soniox(language_hints, language_hints_strict, enable_language_identification)
    if provider == "sarvam":
        return _create_sarvam()
    if provider == "gnani":
        return _create_gnani()

    raise ValueError(
        f"[STT Factory] Unsupported STT_PROVIDER='{provider}'. "
        "Valid values: soniox, sarvam, gnani"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Soniox
# ──────────────────────────────────────────────────────────────────────────────

# Domain vocabulary hint for Soniox's native context biasing (context_version 2,
# stt-rt-v3-preview and higher — stt-rt-v5 qualifies). Fixes brand-name mishearing
# (e.g. STT hearing "ఎగ్ రిజల్ట్స్" instead of the company name) at the ASR level
# instead of trying to fuzzy-correct the transcript after the fact, which fails
# on mishearings that change word count/character overlap.
#
# Shared with transcript_correction.py's post-STT dictionary fix — see
# providers/domain_vocab.py for the single source of truth.
from .domain_vocab import CONTEXT_TEXT as _SONIOX_CONTEXT_TEXT, DOMAIN_TERMS as _SONIOX_CONTEXT_TERMS


def _create_soniox(
    language_hints: Optional[List[Language]],
    language_hints_strict: bool,
    enable_language_identification: bool,
):
    from pipecat.services.soniox.stt import SonioxSTTService, SonioxContextObject
    from pipecat.services.stt_service import STTService
    from pipecat.frames.frames import StartFrame
    from websockets.protocol import State

    class _FastSonioxSTTService(SonioxSTTService):
        # SonioxSTTService.start() awaits the full websocket handshake
        # (~1.5-2.3s) before returning, which blocks this processor's frame
        # queue — including any frame queued right behind StartFrame, like
        # the greeting's TTSSpeakFrame — until Soniox finishes connecting.
        # We call the grandparent's start() directly (sets _sample_rate from
        # the StartFrame, same as stock behavior) and kick off _connect() as
        # a background task instead, so StartFrame (and the greeting behind
        # it) flow downstream immediately. The handshake normally finishes
        # well within the ~9s greeting playback, long before the caller's
        # first turn needs transcribing — EXCEPT when a caller (typically a
        # returning one who already knows the flow) barges in within the
        # first ~1.5-2.3s of the call, before the handshake completes. Stock
        # run_stt() just no-ops on any audio that arrives while the socket
        # isn't OPEN yet, silently dropping it — confirmed live 2026-07-29
        # (call_919154315557_fbf271d5.log: caller started speaking 18ms
        # before Soniox reported connected, her whole 1.2s utterance was
        # lost, the turn timed out with an empty transcript, and she hung up
        # into the resulting silence). We buffer instead of dropping, and
        # flush the buffer the moment the socket opens.
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._pending_audio: list[bytes] = []

        async def start(self, frame: StartFrame):
            await STTService.start(self, frame)

            connect_task = asyncio.create_task(self._connect())

            def _on_connect_done(task: "asyncio.Task") -> None:
                if task.cancelled():
                    return
                exc = task.exception()
                if exc is not None:
                    logger.warning("Soniox STT background connect failed", exc_info=exc)

            connect_task.add_done_callback(_on_connect_done)

        async def run_stt(self, audio: bytes):
            if not (self._websocket and self._websocket.state is State.OPEN):
                self._pending_audio.append(audio)
                yield None
                return

            if self._pending_audio:
                buffered, self._pending_audio = self._pending_audio, []
                logger.debug(
                    f"Soniox STT: flushing {len(buffered)} audio chunk(s) buffered "
                    "during connect handshake"
                )
                for chunk in buffered:
                    try:
                        await self._websocket.send(chunk)
                    except Exception as e:
                        logger.warning(f"{self}: buffered send failed: {e}")

            async for out in super().run_stt(audio):
                yield out

    api_key = os.getenv("SONIOX_API_KEY")
    if not api_key:
        raise ValueError("[STT Factory] SONIOX_API_KEY is not set in .env")

    model = os.getenv("SONIOX_MODEL", "stt-rt-v5")

    ttfs_p99_latency   = float(os.getenv("SONIOX_TTFS_P99_LATENCY", "0.35"))
    enable_diarization = os.getenv("SONIOX_ENABLE_DIARIZATION", "false").lower() == "true"

    context = SonioxContextObject(text=_SONIOX_CONTEXT_TEXT, terms=_SONIOX_CONTEXT_TERMS)

    # endpoint_sensitivity / max_endpoint_delay_ms are Soniox-side endpoint
    # detection fields and are only valid when enable_endpoint_detection=True.
    # We use vad_force_turn_endpoint=True (local Silero VAD) instead, so
    # Soniox's own endpoint detection is off — these fields must NOT be sent.
    logger.info(
        f"[STT Factory] Soniox model={model} ttfs_p99_latency={ttfs_p99_latency} "
        f"context_terms={len(_SONIOX_CONTEXT_TERMS)}"
    )

    return _FastSonioxSTTService(
        api_key=api_key,
        vad_force_turn_endpoint=True,
        ttfs_p99_latency=ttfs_p99_latency,
        settings=SonioxSTTService.Settings(
            model=model,
            language_hints=language_hints,
            language_hints_strict=language_hints_strict,
            enable_language_identification=enable_language_identification,
            enable_speaker_diarization=enable_diarization,
            context=context,
        ),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Sarvam
# ──────────────────────────────────────────────────────────────────────────────

def _create_sarvam():
    try:
        from pipecat.services.sarvam.stt import SarvamSTTService
    except ImportError as exc:
        raise ImportError(
            "[STT Factory] Sarvam STT service not found in installed pipecat version. "
            "Check pipecat release notes or install pipecat[sarvam]."
        ) from exc

    api_key = os.getenv("SARVAM_STT_API_KEY")
    if not api_key:
        raise ValueError("[STT Factory] SARVAM_STT_API_KEY is not set in .env")

    model = os.getenv("SARVAM_STT_MODEL", "saaras:v3")
    logger.info(f"[STT Factory] Sarvam STT model={model}")

    return SarvamSTTService(
        api_key=api_key,
        settings=SarvamSTTService.Settings(model=model),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Gnani.ai Prisma v2.5  (uses official pipecat_gnani plugin)
# ──────────────────────────────────────────────────────────────────────────────

# Maps GNANI_STT_LANGUAGE values → pipecat Language enum (Gnani needs IN-suffixed codes)
_GNANI_LANG_MAP = {
    "te": Language.TE_IN, "te-IN": Language.TE_IN,
    "hi": Language.HI_IN, "hi-IN": Language.HI_IN,
    "ta": Language.TA_IN, "ta-IN": Language.TA_IN,
    "kn": Language.KN_IN, "kn-IN": Language.KN_IN,
    "ml": Language.ML_IN, "ml-IN": Language.ML_IN,
    "mr": Language.MR_IN, "mr-IN": Language.MR_IN,
    "gu": Language.GU_IN, "gu-IN": Language.GU_IN,
    "bn": Language.BN_IN, "bn-IN": Language.BN_IN,
    "pa": Language.PA_IN, "pa-IN": Language.PA_IN,
    "en": Language.EN_IN, "en-IN": Language.EN_IN,
}

def _create_gnani():
    from pipecat_gnani.stt import GnaniSTTService

    api_key  = os.getenv("GNANI_API_KEY")
    if not api_key:
        raise ValueError("[STT Factory] GNANI_API_KEY is not set in .env")

    lang_raw = os.getenv("GNANI_STT_LANGUAGE", "en-IN").lower().strip()
    language = _GNANI_LANG_MAP.get(lang_raw, Language.EN_IN)

    logger.info(f"[STT Factory] Gnani STT (pipecat_gnani)  language={lang_raw}→{language}")

    class _GnaniAutoReconnect(GnaniSTTService):
        async def _receive_messages(self):
            import asyncio
            import json as _json
            from pipecat.frames.frames import ErrorFrame, TranscriptionFrame
            from pipecat.utils.time import time_now_iso8601
            from pipecat_gnani._common import get_language_string, stt_language_to_gnani

            try:
                async for msg in self._ws:
                    if isinstance(msg, bytes):
                        continue
                    data = _json.loads(msg)
                    msg_type = data.get("type", "")

                    if msg_type == "transcript":
                        text = data.get("text", "")
                        if text:
                            lang = get_language_string(self._settings, stt_language_to_gnani)
                            await self.push_frame(TranscriptionFrame(
                                text=text,
                                user_id=self._user_id if hasattr(self, "_user_id") else "",
                                timestamp=time_now_iso8601(),
                                language=lang,
                            ))

                    elif msg_type == "error":
                        error_msg = data.get("message", "Unknown error")
                        if "no speech segment" in error_msg.lower():
                            # Expected idle-timeout — downgrade to DEBUG, no ErrorFrame
                            logger.debug(f"[Gnani STT] idle timeout: {error_msg}")
                        else:
                            logger.error(f"[Gnani STT] stream error: {error_msg}")
                            await self.push_frame(ErrorFrame(error=f"Gnani STT: {error_msg}"))

                    # processing / speech_start / vad_start / speech_end / vad_end → ignore

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.debug(f"[Gnani STT] receive loop ended: {e}")
            finally:
                self._ws = None
                logger.debug("[Gnani STT] session closed — will reconnect on next audio")

        async def run_stt(self, audio):
            if not self._ws:
                logger.info("[Gnani STT] reconnecting dropped session...")
                await self._connect()
            if self._ws:
                try:
                    await self._ws.send(audio)
                except Exception:
                    self._ws = None  # dead socket, next call will reconnect
            yield None

    return _GnaniAutoReconnect(
        api_key=api_key,
        # model=None + sample_rate=None silence the NOT_GIVEN validation warnings
        settings=GnaniSTTService.Settings(language=language, model=None, sample_rate=None),
    )
