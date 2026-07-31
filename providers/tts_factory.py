"""
TTS Factory — reads TTS_PROVIDER from .env and returns the configured service.

Supported providers (set TTS_PROVIDER= in .env):
    cartesia  → CartesiaTTSService   (default)
    sarvam    → SarvamTTSService
    xai/grok  → XAITTSService        (Grok / xAI WebSocket streaming)
    bakbak    → BakbakTTSService     (custom HTTP client — not in pipecat)
    soniox    → SonioxTTSService     (WebSocket streaming, 60+ languages)
    murf      → MurfTTSService       (pipecat-murf-tts pkg, WebSocket, Falcon 2)

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

    raise ValueError(
        f"[TTS Factory] Unsupported TTS_PROVIDER='{provider}'. "
        "Valid values: cartesia, sarvam, xai, grok, bakbak, soniox, murf"
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

    # Only model available: tts-rt-v1
    model = os.getenv("SONIOX_TTS_MODEL", "tts-rt-v1")

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
