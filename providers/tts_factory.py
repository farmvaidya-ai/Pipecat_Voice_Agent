"""
TTS Factory — reads TTS_PROVIDER from .env and returns the configured service.

Supported providers (set TTS_PROVIDER= in .env):
    cartesia  → CartesiaTTSService   (default)
    sarvam    → SarvamTTSService
    xai/grok  → XAITTSService        (Grok / xAI WebSocket streaming)
    bakbak    → BakbakTTSService     (custom HTTP client — not in pipecat)
    soniox    → SonioxTTSService     (WebSocket streaming, 60+ languages)
    murf      → MurfTTSService       (pipecat-murf-tts pkg, WebSocket, Falcon 2)
    svara     → SvaraTTSService      (custom HTTP client — svara-voice SDK's
                                       own pipecat wrapper targets an older
                                       TTSService contract, see providers/svara_tts.py)
    elevenlabs → ElevenLabsTTSService (official pipecat integration, WebSocket
                                       streaming, model=eleven_flash_v2_5 by default)

Adding a new provider:
    1. Add its env vars to .env (e.g. NEWPROVIDER_TTS_API_KEY=, NEWPROVIDER_TTS_MODEL=)
    2. Add an elif branch in create_tts() that calls a new _create_newprovider_tts() helper
    3. Write the helper function below following the same pattern
    4. Update the error message in create_tts() to list the new provider
"""

import os
from typing import Optional

from loguru import logger
from pipecat.transcriptions.language import Language


def create_tts(
    voice_id: Optional[str] = None,
    language: Optional[Language] = None,
):
    """
    Instantiate and return the TTS service selected by TTS_PROVIDER in .env.

    Args:
        voice_id:  Cartesia voice UUID (or Sarvam voice name). Falls back to
                   CARTESIA_VOICE_ID / SARVAM_TTS_VOICE env var if not supplied.
        language:  Initial TTS output language. Falls back to Language.EN if
                   not supplied and can't be determined from env.

    Returns:
        Configured TTS service instance ready for pipeline use.

    Raises:
        ValueError:  Required env var is missing.
        ImportError: Provider package is not installed.
        ValueError:  TTS_PROVIDER names an unsupported provider.
    """
    provider = os.getenv("TTS_PROVIDER", "cartesia").lower().strip()
    logger.info(f"[TTS Factory] provider={provider.upper()}")

    if provider == "cartesia":
        return _create_cartesia(voice_id, language)
    if provider == "sarvam":
        return _create_sarvam(voice_id, language)
    if provider in ("xai", "grok"):
        return _create_xai(voice_id, language)
    if provider == "bakbak":
        return _create_bakbak(voice_id, language)
    if provider == "soniox":
        return _create_soniox_tts(voice_id, language)
    if provider == "murf":
        return _create_murf(voice_id, language)
    if provider == "svara":
        return _create_svara(voice_id, language)
    if provider == "elevenlabs":
        return _create_elevenlabs(voice_id, language)

    raise ValueError(
        f"[TTS Factory] Unsupported TTS_PROVIDER='{provider}'. "
        "Valid values: cartesia, sarvam, xai, grok, bakbak, soniox, murf, svara, elevenlabs"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Cartesia
# ──────────────────────────────────────────────────────────────────────────────

def _create_cartesia(voice_id: Optional[str], language: Optional[Language]):
    from pipecat.services.cartesia.tts import CartesiaTTSService

    api_key = os.getenv("CARTESIA_API_KEY")
    if not api_key:
        raise ValueError("[TTS Factory] CARTESIA_API_KEY is not set in .env")

    model = os.getenv("CARTESIA_MODEL", "sonic-3.5")

    # Resolve voice: caller > CARTESIA_VOICE_ID env > hard fail (must be set)
    resolved_voice = voice_id or os.getenv("CARTESIA_VOICE_ID") or ""
    if not resolved_voice:
        raise ValueError(
            "[TTS Factory] No voice_id supplied and CARTESIA_VOICE_ID is not set in .env. "
            "Set CARTESIA_VOICE_ID to a Cartesia voice UUID."
        )

    effective_language = language or Language.EN

    logger.info(
        f"[TTS Factory] Cartesia model={model} "
        f"voice={resolved_voice} "
        f"language={effective_language}"
    )

    return CartesiaTTSService(
        api_key=api_key,
        settings=CartesiaTTSService.Settings(
            model=model,
            voice=resolved_voice,
            language=effective_language,
        ),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Sarvam
# ──────────────────────────────────────────────────────────────────────────────

def _create_sarvam(voice_id: Optional[str], language: Optional[Language]):
    try:
        from pipecat.services.sarvam.tts import SarvamTTSService
    except ImportError as exc:
        raise ImportError(
            "[TTS Factory] Sarvam TTS service not found in installed pipecat version. "
            "Check pipecat release notes or install pipecat[sarvam]."
        ) from exc

    api_key = os.getenv("SARVAM_TTS_API_KEY")
    if not api_key:
        raise ValueError("[TTS Factory] SARVAM_TTS_API_KEY is not set in .env")

    # bulbul:v1 was removed by Sarvam; default to v3
    model = os.getenv("SARVAM_TTS_MODEL", "bulbul:v3")

    # Sarvam uses a speaker name, not a UUID. Always read from SARVAM_TTS_VOICE —
    # the voice_id param may carry a Cartesia UUID when switching providers, so
    # it is intentionally ignored here.
    resolved_voice = os.getenv("SARVAM_TTS_VOICE", "anushka")

    # Startup language for Sarvam TTS.
    # SARVAM_TTS_LANGUAGE in .env sets the boot language (e.g. "en-IN", "te-IN").
    # Leave it unset or set to "auto" to let Sarvam default to en-IN and let
    # MultilingualTTSSwitcher take over after the first detected utterance.
    _lang_env = os.getenv("SARVAM_TTS_LANGUAGE", "").strip().lower()
    # Sarvam TTS requires a valid BCP-47 string — it rejects None.
    # "auto" / unset → start with "en-IN"; MultilingualTTSSwitcher updates it
    # after the first detected utterance.
    startup_language: str = "en-IN" if (_lang_env in ("", "auto")) else _lang_env

    logger.info(
        f"[TTS Factory] Sarvam TTS model={model} voice={resolved_voice} "
        f"startup_language={startup_language or 'auto (en-IN default)'}"
    )

    return SarvamTTSService(
        api_key=api_key,
        settings=SarvamTTSService.Settings(
            model=model,
            voice=resolved_voice,
            language=startup_language,
        ),
    )


# ──────────────────────────────────────────────────────────────────────────────
# xAI / Grok
# ──────────────────────────────────────────────────────────────────────────────

def _create_xai(_voice_id: Optional[str], language: Optional[Language]):
    try:
        from pipecat.services.xai.tts import XAITTSService
    except ImportError as exc:
        raise ImportError(
            "[TTS Factory] xAI TTS service not found. "
            "Run: pip install pipecat-ai[xai]"
        ) from exc

    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise ValueError("[TTS Factory] XAI_API_KEY is not set in .env")

    # xAI voices: eve, ember, aurora, cove, dusk, breeze
    # "cove" is a natural-sounding male voice (set XAI_TTS_VOICE to override)
    resolved_voice = os.getenv("XAI_TTS_VOICE", "sal")  # "sal" is best overall multilingual voice
    resolved_model = os.getenv("XAI_TTS_MODEL") or None   # None = xAI default
    _lang_env = os.getenv("XAI_TTS_LANGUAGE", "").strip().lower()
    effective_language = language or (Language.EN if _lang_env in ("", "auto") else _lang_env)

    logger.info(
        f"[TTS Factory] xAI TTS voice={resolved_voice} model={resolved_model or 'default'} "
        f"language={effective_language}"
    )

    return XAITTSService(
        api_key=api_key,
        settings=XAITTSService.Settings(
            model=resolved_model,
            voice=resolved_voice,
            language=effective_language,
        ),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Bakbak
# ──────────────────────────────────────────────────────────────────────────────

def _create_bakbak(_voice_id: Optional[str], _language: Optional[Language]):
    from providers.bakbak_tts import BakbakTTSService

    api_key = os.getenv("BAKBAK_API_KEY")
    if not api_key:
        raise ValueError("[TTS Factory] BAKBAK_API_KEY is not set in .env")

    # Bakbak identifies voices by UUID. Raya's ID is the default.
    resolved_voice_id   = os.getenv("BAKBAK_TTS_VOICE_ID", "d6a002d0-230c-49b1-a137-b8a7d564b1ae")
    resolved_voice_name = os.getenv("BAKBAK_TTS_VOICE_NAME", "Raya")

    # "standard" is Bakbak's default model tier
    model = os.getenv("BAKBAK_TTS_MODEL", "standard")

    # Language: Bakbak uses short ISO codes (en, hi, te, …)
    _lang_env = os.getenv("BAKBAK_TTS_LANGUAGE", "").strip().lower()
    startup_language: str = "en" if _lang_env in ("", "auto") else _lang_env

    base_url = os.getenv("BAKBAK_TTS_API_URL", "https://hub.getraya.app/v1/text-to-speech")

    # "pcm" skips the AAC decode step entirely (Bakbak returns raw PCM
    # instead of ADTS AAC) — faster and cheaper per chunk. Kept configurable
    # in case AAC is ever needed again.
    codec = os.getenv("BAKBAK_CODEC", "pcm").strip().lower()

    logger.info(
        f"[TTS Factory] Bakbak TTS voice={resolved_voice_name} ({resolved_voice_id}) "
        f"model={model} language={startup_language} codec={codec} "
        f"endpoint={base_url}"
    )

    max_chars = int(os.getenv("BAKBAK_TTS_MAX_CHARS", "60"))

    # Word-batch streaming: unset/0 = off (default sentence-level TTS).
    _word_batch_env = os.getenv("BAKBAK_TTS_WORD_BATCH_SIZE", "").strip()
    word_batch_size = int(_word_batch_env) if _word_batch_env else None
    clause_lookahead_words = int(os.getenv("BAKBAK_TTS_CLAUSE_LOOKAHEAD_WORDS", "3"))

    if word_batch_size:
        logger.info(
            f"[TTS Factory] Bakbak word-batch streaming ENABLED: "
            f"flush every {word_batch_size} words (clause lookahead {clause_lookahead_words})"
        )

    return BakbakTTSService(
        api_key=api_key,
        voice_id=resolved_voice_id,
        language=startup_language,
        model=model,
        codec=codec,
        base_url=base_url,
        max_chars=max_chars,
        word_batch_size=word_batch_size,
        clause_lookahead_words=clause_lookahead_words,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Soniox RT TTS
# ──────────────────────────────────────────────────────────────────────────────

def _create_soniox_tts(_voice_id: Optional[str], language: Optional[Language]):
    try:
        from pipecat.services.soniox.tts import SonioxTTSService
    except ImportError as exc:
        raise ImportError(
            "[TTS Factory] Soniox TTS service not found in installed pipecat. "
            "Run: pip install pipecat-ai[soniox]"
        ) from exc

    # SONIOX_TTS_API_KEY takes priority; falls back to the shared SONIOX_API_KEY
    api_key = os.getenv("SONIOX_TTS_API_KEY") or os.getenv("SONIOX_API_KEY")
    if not api_key:
        raise ValueError(
            "[TTS Factory] Neither SONIOX_TTS_API_KEY nor SONIOX_API_KEY is set in .env"
        )

    # tts-rt-v2 is the current production model (launched 2026-08): better voice
    # quality, audio-tag emotion/delivery control, voice cloning, in-sentence
    # language switching. Fully backward-compatible with tts-rt-v1's API/settings
    # (model name is the only thing that changes). tts-rt-v1 is deprecated and
    # will be removed 2026-08-31, after which it auto-routes to v2 anyway.
    model = os.getenv("SONIOX_TTS_MODEL", "tts-rt-v2")

    # Voice name — see console.soniox.com for full gallery
    resolved_voice = os.getenv("SONIOX_TTS_VOICE", "Adrian")

    # Language: ISO code (en, hi, te, ta, kn, ml, …) or "auto" → EN startup,
    # then MultilingualTTSSwitcher switches per detected language at runtime.
    _lang_env = os.getenv("SONIOX_TTS_LANGUAGE", "").strip().lower()
    effective_language = language or (Language.EN if _lang_env in ("", "auto") else _lang_env)

    logger.info(
        f"[TTS Factory] Soniox TTS model={model} voice={resolved_voice} "
        f"startup_language={effective_language}"
    )

    return SonioxTTSService(
        api_key=api_key,
        settings=SonioxTTSService.Settings(
            model=model,
            voice=resolved_voice,
            language=effective_language,
        ),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Murf AI
# ──────────────────────────────────────────────────────────────────────────────

def _create_murf(_voice_id: Optional[str], _language: Optional[Language]):
    try:
        from pipecat_murf_tts import MurfTTSService
    except ImportError as exc:
        raise ImportError(
            "[TTS Factory] pipecat-murf-tts not installed. Run: pip install pipecat-murf-tts"
        ) from exc

    api_key = os.getenv("MURF_API_KEY")
    if not api_key:
        raise ValueError("[TTS Factory] MURF_API_KEY is not set in .env")

    # en-IN-samar (male) is the default startup voice. Note GET /v1/speech/voices
    # needs ?model=FALCON to surface this — the default catalog response is a
    # different/smaller list that omits it. bot_processors/tts_switcher.py
    # swaps BOTH voice_id and locale per detected language
    # (_LANG_TO_MURF_VOICE) — native per-language voice where Falcon-2 has
    # one, else en-IN-samar cross-lingually steered via multi_native_locale.
    # _voice_id may carry a leftover Cartesia UUID from Bot.py's shared
    # VOICE_ID env var — intentionally ignored here, same as Sarvam/Bakbak/Soniox.
    resolved_voice = os.getenv("MURF_TTS_VOICE_ID") or "en-IN-samar"

    model = os.getenv("MURF_TTS_MODEL", "falcon-2")
    style = os.getenv("MURF_TTS_STYLE", "Conversational")
    rate = int(os.getenv("MURF_TTS_RATE", "0"))
    pitch = int(os.getenv("MURF_TTS_PITCH", "0"))
    sample_rate = int(os.getenv("MURF_TTS_SAMPLE_RATE", "24000"))
    channel_type = os.getenv("MURF_TTS_CHANNEL_TYPE", "MONO")
    audio_format = os.getenv("MURF_TTS_FORMAT", "PCM")

    # Startup cross-lingual locale (e.g. "te-IN"). Blank/"auto" = no locale
    # override, voice speaks in its own native locale (en-US for natalie).
    _locale_env = os.getenv("MURF_TTS_LOCALE", "").strip()
    resolved_locale = None if _locale_env in ("", "auto") else _locale_env

    logger.info(
        f"[TTS Factory] Murf TTS voice={resolved_voice} model={model} style={style} "
        f"locale={resolved_locale or 'native'} sample_rate={sample_rate} format={audio_format}"
    )

    return MurfTTSService(
        api_key=api_key,
        params=MurfTTSService.InputParams(
            voice_id=resolved_voice,
            model=model,
            style=style,
            rate=rate,
            pitch=pitch,
            sample_rate=sample_rate,
            channel_type=channel_type,
            format=audio_format,
            multi_native_locale=resolved_locale,
        ),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Svara (Kenpath Labs)
# ──────────────────────────────────────────────────────────────────────────────

def _create_svara(_voice_id: Optional[str], _language: Optional[Language]):
    try:
        from providers.svara_tts import SvaraTTSService
    except ImportError as exc:
        raise ImportError(
            "[TTS Factory] svara-voice not installed. Run: "
            "pip install git+https://github.com/kenpath-labs/svara-python.git"
        ) from exc

    api_key = os.getenv("SVARA_API_KEY")
    if not api_key:
        raise ValueError("[TTS Factory] SVARA_API_KEY is not set in .env")

    # Optional — defaults to https://api.kenpathlabs.com inside AsyncSvara if unset.
    # Must be the bare origin, NOT ending in /v1 — AsyncSvara's own internal
    # calls already target paths like /v1/audio/speech, and httpx's base_url
    # merging APPENDS rather than replacing on a leading "/" (unlike plain
    # urljoin), so a base_url of ".../v1" plus "/v1/audio/speech" silently
    # doubles up into ".../v1/v1/audio/speech" and 404s. Stripped defensively
    # here in case .env ever gets a doc-pasted URL with /v1 on the end again.
    base_url = (os.getenv("SVARA_BASE_URL") or "").strip().rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[: -len("/v1")]
    base_url = base_url or None

    # _voice_id may carry a leftover Cartesia UUID from Bot.py's shared
    # VOICE_ID env var — intentionally ignored here, same as Sarvam/Bakbak/Murf.
    resolved_voice = os.getenv("SVARA_TTS_VOICE", "sv_enhdbrj5")  # Aanya (Hindi native, speaks all 80 langs)
    # Display-only, for the startup log line below — same VOICE_ID/VOICE_NAME
    # pairing Bakbak uses, since Svara's own IDs (sv_xxxxxxxx) aren't readable
    # on their own. Not sent to the API; keep in sync by hand if you change
    # SVARA_TTS_VOICE (see docs/svara_voice_roster.pdf for the full roster).
    resolved_voice_name = os.getenv("SVARA_TTS_VOICE_NAME", "")

    model = os.getenv("SVARA_TTS_MODEL", "svara-1")

    # Any of mp3/opus/aac/flac/wav/pcm/ulaw/alaw — pcm is headerless s16le,
    # cheapest to stream straight into the pipeline (see providers/svara_tts.py).
    response_format = os.getenv("SVARA_TTS_FORMAT", "pcm")

    _speed_env = os.getenv("SVARA_TTS_SPEED", "").strip()
    speed = None
    if _speed_env:
        try:
            speed = float(_speed_env)
        except ValueError:
            logger.warning(
                f"[TTS Factory] SVARA_TTS_SPEED={_speed_env!r} is not a valid number, "
                "falling back to Svara's server default (1.0)"
            )

    # Startup language hint (ISO code, e.g. "te", "hi") — "auto"/unset lets
    # Svara auto-detect from the text itself. MultilingualTTSSwitcher updates
    # this per-turn once a caller's language is detected (see tts_switcher.py).
    _lang_env = os.getenv("SVARA_TTS_LANGUAGE", "").strip().lower()
    startup_language = None if _lang_env in ("", "auto") else _lang_env

    _voice_display = f"{resolved_voice_name} ({resolved_voice})" if resolved_voice_name else resolved_voice
    logger.info(
        f"[TTS Factory] Svara TTS voice={_voice_display} model={model} "
        f"format={response_format} startup_language={startup_language or 'auto'}"
    )

    return SvaraTTSService(
        api_key=api_key,
        base_url=base_url,
        voice=resolved_voice,
        model=model,
        language=startup_language,
        response_format=response_format,
        speed=speed,
    )


# ──────────────────────────────────────────────────────────────────────────────
# ElevenLabs
# ──────────────────────────────────────────────────────────────────────────────

def _create_elevenlabs(_voice_id: Optional[str], language: Optional[Language]):
    try:
        from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
    except ImportError as exc:
        raise ImportError(
            "[TTS Factory] ElevenLabs TTS service not found in installed pipecat. "
            "Run: pip install pipecat-ai[elevenlabs]"
        ) from exc

    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise ValueError("[TTS Factory] ELEVENLABS_API_KEY is not set in .env")

    # eleven_flash_v2_5: lowest-latency ElevenLabs model (~75ms), multilingual
    # (supports language_code — see ELEVENLABS_MULTILINGUAL_MODELS in
    # pipecat's elevenlabs/tts.py; eleven_flash_v2 and eleven_turbo_v2 do NOT).
    model = os.getenv("ELEVENLABS_TTS_MODEL", "eleven_flash_v2_5")

    # ElevenLabs voice IDs are account-specific (no public default like
    # Bakbak/Soniox). _voice_id may carry a leftover Cartesia UUID from
    # Bot.py's shared VOICE_ID env var — intentionally ignored here, same as
    # Sarvam/Bakbak/Murf/Svara — always read from ELEVENLABS_TTS_VOICE_ID.
    resolved_voice = os.getenv("ELEVENLABS_TTS_VOICE_ID") or ""
    if not resolved_voice:
        raise ValueError(
            "[TTS Factory] No voice_id supplied and ELEVENLABS_TTS_VOICE_ID is not set "
            "in .env. Set ELEVENLABS_TTS_VOICE_ID to a voice ID from your ElevenLabs "
            "account (Voice Library → copy Voice ID)."
        )
    # Display-only, for the startup log line below — ElevenLabs' own IDs
    # aren't readable on their own, same VOICE_ID/VOICE_NAME pairing
    # Bakbak/Svara use. Not sent to the API; keep in sync by hand if you
    # change ELEVENLABS_TTS_VOICE_ID.
    resolved_voice_name = os.getenv("ELEVENLABS_TTS_VOICE_NAME", "")

    # Language: ISO code (en, hi, ta — ElevenLabs' language_code list only
    # covers these three among this project's Indian languages, see
    # ELEVENLABS_MULTILINGUAL_MODELS in pipecat/services/elevenlabs/tts.py)
    # or "auto" → EN startup, then MultilingualTTSSwitcher switches per
    # detected language at runtime (falling back to EN for unsupported ones).
    _lang_env = os.getenv("ELEVENLABS_TTS_LANGUAGE", "").strip().lower()
    effective_language = language or (Language.EN if _lang_env in ("", "auto") else _lang_env)

    # Optional voice-character knobs — left NOT_GIVEN (ElevenLabs server
    # defaults) unless explicitly set in .env.
    def _opt_float(name: str) -> Optional[float]:
        raw = os.getenv(name, "").strip()
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            logger.warning(f"[TTS Factory] {name}={raw!r} is not a valid number, ignoring")
            return None

    stability = _opt_float("ELEVENLABS_TTS_STABILITY")
    similarity_boost = _opt_float("ELEVENLABS_TTS_SIMILARITY_BOOST")
    speed = _opt_float("ELEVENLABS_TTS_SPEED")

    settings_kwargs = {
        "model": model,
        "voice": resolved_voice,
        "language": effective_language,
    }
    if stability is not None:
        settings_kwargs["stability"] = stability
    if similarity_boost is not None:
        settings_kwargs["similarity_boost"] = similarity_boost
    if speed is not None:
        settings_kwargs["speed"] = speed

    _voice_display = f"{resolved_voice_name} ({resolved_voice})" if resolved_voice_name else resolved_voice
    logger.info(
        f"[TTS Factory] ElevenLabs TTS voice={_voice_display} model={model} "
        f"language={effective_language}"
    )

    return ElevenLabsTTSService(
        api_key=api_key,
        settings=ElevenLabsTTSService.Settings(**settings_kwargs),
    )
