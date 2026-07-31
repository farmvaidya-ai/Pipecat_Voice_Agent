"""
Provider factories — import these in cat.py instead of service classes directly.

Usage:
    from providers import create_stt, create_tts, create_llm

    stt = create_stt(language_hints=_ALL_LANGS)
    tts = create_tts(voice_id=VOICE_ID, language=TTS_INITIAL_LANG)
    llm = create_llm()

Provider selection is driven entirely by .env:
    STT_PROVIDER=soniox | sarvam
    TTS_PROVIDER=cartesia | sarvam
    LLM_PROVIDER=vertex | openai | groq
"""

from .stt_factory import create_stt
from .tts_factory import create_tts
from .llm_factory import create_llm

__all__ = ["create_stt", "create_tts", "create_llm"]
