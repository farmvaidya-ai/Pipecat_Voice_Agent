"""Multilingual TTS language switcher (LANGUAGE=auto mode only)."""

from loguru import logger

from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transcriptions.language import Language

_SUPPORTED_TTS_LANGS = {
    Language.TE, Language.HI, Language.TA, Language.KN, Language.ML, Language.EN,
    Language.MR, Language.BN, Language.GU, Language.PA,
}
# Odia and Assamese deliberately excluded here: number-word conversion exists
# for them (_NUMBER_LANG_CONFIG) but Bakbak has no voice for either, so
# switching the active TTS language to OR/AS would produce mispronounced
# English-voiced audio. Any OR/AS transcription falls back to Language.EN below.

# Murf (via en-IN-samar cross-lingual steering) DOES have real or-IN/as-IN
# coverage, unlike Bakbak — so Murf gets its own wider supported-langs set.
_MURF_SUPPORTED_TTS_LANGS = _SUPPORTED_TTS_LANGS | {Language.OR, Language.AS}

# Sarvam TTS requires BCP-47 strings (e.g. "en-IN"), not Language enum objects.
# Cartesia accepts Language enums directly.
_LANG_TO_SARVAM_CODE: dict[Language, str] = {
    Language.EN: "en-IN",
    Language.HI: "hi-IN",
    Language.TE: "te-IN",
    Language.TA: "ta-IN",
    Language.KN: "kn-IN",
    Language.ML: "ml-IN",
}

# Murf falcon-2 has no exposed public per-turn "language" setter, but its
# voice_id CAN be swapped via _settings.voice (read fresh in
# _build_voice_config_message), and multi_native_locale can be swapped via
# the private _murf_settings dict (also read fresh per-call). So "switching
# language" for Murf means swapping BOTH to the best available combo per
# GET /v1/speech/voices?model=FALCON (note the ?model=FALCON — the default,
# unfiltered response is a different/smaller catalog missing several of
# these voices entirely).
#
# Native per-language male voice used where Falcon-2 has one; the four
# locales with no native voice at all (te-IN, kn-IN, as-IN, or-IN) fall back
# to en-IN-samar cross-lingually steered via multi_native_locale. Only
# en-IN-samar + te-IN has been human-verified by ear so far (2026-07-23);
# the rest are per Murf's own supportedLocales/catalog metadata.
_LANG_TO_MURF_VOICE: dict[Language, tuple[str, str]] = {
    Language.EN: ("en-IN-samar", "en-IN"),
    Language.HI: ("hi-IN-aman", "hi-IN"),
    Language.TE: ("en-IN-samar", "te-IN"),       # no native te-IN voice on this account
    Language.TA: ("ta-IN-karthikeyan", "ta-IN"),
    Language.KN: ("en-IN-samar", "kn-IN"),       # no native kn-IN male voice
    Language.ML: ("ml-IN-madhavan", "ml-IN"),
    Language.MR: ("mr-IN-vaibhav", "mr-IN"),
    Language.BN: ("bn-IN-subhankar", "bn-IN"),
    Language.GU: ("gu-IN-hardik", "gu-IN"),
    Language.PA: ("pa-IN-harman", "pa-IN"),
    Language.OR: ("en-IN-samar", "or-IN"),       # no native or-IN voice on this account
    Language.AS: ("en-IN-samar", "as-IN"),       # no native as-IN voice on this account
}


class MultilingualTTSSwitcher(FrameProcessor):
    """Watches detected language on TranscriptionFrame and hot-swaps TTS language.

    Handles provider differences:
    - Cartesia : sets _settings.language to a Language enum
    - Sarvam   : sets _settings.language to a BCP-47 string ("en-IN"),
                 because the Sarvam API rejects raw Language enum values
    """

    def __init__(self, tts_service, **kwargs):
        super().__init__(**kwargs)
        self._tts = tts_service
        self._current_lang = None

        # Detect provider at construction time so we don't import on every frame
        try:
            from pipecat.services.sarvam.tts import SarvamTTSService
            self._is_sarvam = isinstance(tts_service, SarvamTTSService)
        except ImportError:
            self._is_sarvam = False

        try:
            from providers.bakbak_tts import BakbakTTSService
            self._is_bakbak = isinstance(tts_service, BakbakTTSService)
        except ImportError:
            self._is_bakbak = False

        try:
            from pipecat_murf_tts import MurfTTSService
            self._is_murf = isinstance(tts_service, MurfTTSService)
        except ImportError:
            self._is_murf = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and frame.language:
            allowed_langs = _MURF_SUPPORTED_TTS_LANGS if self._is_murf else _SUPPORTED_TTS_LANGS
            new_lang = frame.language if frame.language in allowed_langs else Language.EN
            if new_lang != self._current_lang:
                self._current_lang = new_lang
                try:
                    if self._is_sarvam:
                        # Sarvam needs a BCP-47 string; enum value causes rejection
                        self._tts._settings.language = _LANG_TO_SARVAM_CODE.get(new_lang, "en-IN")
                    elif self._is_bakbak:
                        # Bakbak uses short ISO codes: en, hi, te, …
                        _LANG_TO_BAKBAK_SHORT = {
                            Language.EN: "en", Language.HI: "hi", Language.TE: "te",
                            Language.TA: "ta", Language.KN: "kn", Language.ML: "ml",
                            Language.MR: "mr", Language.BN: "bn", Language.GU: "gu", Language.PA: "pa",
                        }
                        self._tts._settings.language = _LANG_TO_BAKBAK_SHORT.get(new_lang, "en")
                    elif self._is_murf:
                        # No public setter for either of these — _settings.voice
                        # is read fresh in _build_voice_config_message, and
                        # _murf_settings is a plain instance dict also read
                        # fresh per-call, so mutating both directly takes
                        # effect next utterance.
                        voice_id, locale = _LANG_TO_MURF_VOICE.get(
                            new_lang, ("en-IN-samar", "en-IN")
                        )
                        self._tts._settings.voice = voice_id
                        self._tts._murf_settings["multi_native_locale"] = locale
                    else:
                        # Cartesia, xAI, and others accept Language enum directly
                        self._tts._settings.language = new_lang
                except Exception as e:
                    logger.warning(f"TTS language switch failed: {e}")
                logger.info(f"🌐 TTS language switched → {new_lang}  (detected: {frame.language})")

        await self.push_frame(frame, direction)
