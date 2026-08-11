"""
Pipecat Voice Agent — Tata Smartflow (Multilingual)
====================================================
Set LANGUAGE in .env: telugu | hindi | tamil | kannada | english | auto

Providers are selected entirely from .env — no hardcoding:
  STT_PROVIDER = soniox | sarvam
  TTS_PROVIDER = cartesia | sarvam
  LLM_PROVIDER = vertex | openai | groq

STT streaming behaviour:
  • Words arrive internally token-by-token while user speaks
    (InterimTranscriptionFrame — visible in logs, NOT sent to LLM)
  • LLM receives ONE complete sentence per turn (TranscriptionFrame)
    emitted ~200–400 ms after speech endpoint is detected
  • There is NO word-by-word trigger to the LLM

Latency logged per turn:
  📝 STT  → full sentence ready (ms after you stop speaking)
  ⚡ LLM  → first response token (ms after STT sentence)
  🔊 TTS  → first audio chunk  (ms after STT sentence)
  🎙️ STT stream → words/sec from interim frames (confirms live streaming)

Cost tracking (TTS):
  Set TTS_PRICE_PER_1K_CHARS in .env (default: 0.065 = $65/million chars)
  💰 logged per turn and as running $/min at call end

Multilingual auto mode:
  When LANGUAGE=auto, TTS language follows detected language from STT.
"""

import asyncio
import contextvars
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*audioop.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*PipelineTask.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*PipelineRunner.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*Passing a worker.*")
import os
import ssl
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service-account.json"

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
ssl._create_default_https_context = ssl._create_unverified_context

# REQUESTS_CA_BUNDLE="" doesn't actually disable SSL in the `requests` library
# (empty string is falsy so requests ignores it). ssl._create_unverified_context
# only patches Python's built-in ssl module, not urllib3/requests.
# google.auth uses requests internally, so we must patch it directly.
import requests as _requests_module
import google.auth.transport.requests as _gatr
_no_ssl_session = _requests_module.Session()
_no_ssl_session.verify = False
_OrigGAuthRequest = _gatr.Request
class _PatchedGAuthRequest(_OrigGAuthRequest):
    def __init__(self, session=None):
        super().__init__(session=session if session is not None else _no_ssl_session)
_gatr.Request = _PatchedGAuthRequest

from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import LLMMessagesAppendFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.turns.user_start.vad_user_turn_start_strategy import VADUserTurnStartStrategy
from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import (
    SpeechTimeoutUserTurnStopStrategy,
)
from pipecat.transcriptions.language import Language
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

from fastapi import FastAPI, Header, Request, WebSocket, WebSocketDisconnect
import uvicorn

from pipecat.services.soniox.stt import SonioxSTTService
from providers.bakbak_tts import BakbakTTSService

# ── Provider factories (reads *_PROVIDER from .env) ──────────────────────────
from providers import create_stt, create_tts, create_llm

# ── RAG retrieval (chonkie + Qdrant knowledge base) ──────────────────────────
from chonkie_rag.search import warmup as rag_warmup

# ── Per-call pipeline processors (split from this file for maintainability) ──
# Positioned after the chonkie_rag import above: chonkie_rag/config.py calls
# load_dotenv() as an import-time side effect, and bot_processors.calls.escalation
# reads os.getenv(...) at module level — it needs that env already loaded.
from bot_processors.calls.serializer import SmartflowFrameSerializer
from bot_processors.voice.latency import _LatencyState, LatencyLogger, CallCostTracker
from bot_processors.calls.escalation import EscalationDetector
from bot_processors.calls.call_ender import CallEndDetector
from bot_processors.voice.number_words import TTSTextSanitizer
from bot_processors.voice.tts_switcher import MultilingualTTSSwitcher
from bot_processors.voice.echo import EchoBuffer, EchoFilter, TranscriptCorrector
from bot_processors.rag.rag import RAGInjector
from bot_processors.core.context_trimmer import ContextTrimmer
from bot_processors.pricing.price_lookup import make_get_price, make_get_price_all_markets
from bot_processors.location.weather_lookup import make_get_weather
from bot_processors.location.location_lookup import (
    make_collect_pincode_digits,
    make_confirm_location,
    make_lookup_place_by_pincode,
    make_save_caller_location,
    make_save_caller_name,
)
from bot_processors.calls.outbound_call import trigger_customer_first_call
from bot_processors.calls.caller_memory import load_latest_summary, note_from_summary, build_template_greeting
from bot_processors.calls.caller_summarizer import summarize_and_save_call, build_llm_greeting
from bot_processors.calls.caller_db import record_call, update_contact_name, get_location
from bot_processors.core.db_pool import init_pool, close_pool
from bot_processors.core.voice_agent_db import init_schema
from bot_processors.calls.call_db import start_call, end_call, save_conversation_messages
from bot_processors.core.task_tracker import track_task

# =========================
# WINDOWS FIX
# =========================
asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# =========================
# LOAD ENV
# =========================
load_dotenv(override=True)

# =========================
# LOGGING
# =========================
# Per-call session_id, propagated via ContextVar rather than passed around
# explicitly. asyncio.to_thread()/create_task() both copy the current context
# by default, so this reaches provider factories (providers/*.py) and pipecat's
# own internals too — not just code lexically inside run_bot().
_current_session_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "_current_session_id", default="-"
)

# What's shown in the log line's [...] tag, keyed by session_id (not a
# ContextVar): _greet_and_inject_memory() sets this from inside its own
# asyncio.create_task(), which gets a *private copy* of the context at fork
# time — a ContextVar.set() there would never be visible to the rest of the
# call (STT, RAG, on_user_turn_started, ...), which run in the original
# context or their own separately-forked tasks. A plain dict mutation is
# visible everywhere immediately, since it isn't context-scoped at all.
# Starts equal to session_id (the caller's number isn't known yet at connect
# time) and is switched to the caller's number once Smartflow's 'start' event
# arrives. Entries are removed in websocket_endpoint's finally block.
_display_id_by_session: dict[str, str] = {}

# asyncio only holds a *weak* reference to Task objects — a fire-and-forget
# asyncio.create_task(...) with no other reference can be garbage-collected
# mid-flight, silently dropping its work (this is how call 3's summary was
# lost). See bot_processors/task_tracker.py, also used by rag.py/latency.py
# for their own background DB writes.
_track_task = track_task


def _inject_session_id(record):
    sid = _current_session_id.get()
    record["extra"]["session_id"] = sid
    record["extra"]["display_id"] = _display_id_by_session.get(sid, sid)


logger.remove(0)
logger.configure(patcher=_inject_session_id)

_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | "
    "<magenta>[{extra[display_id]}]</magenta> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)

logger.add(
    sys.stderr,
    level="DEBUG",
    format=_LOG_FORMAT,
    filter={
        "__main__": "DEBUG",
        "pipecat.services": "DEBUG",
        "pipecat.transports": "INFO",
        "pipecat.pipeline": "INFO",
        "pipecat.processors": "INFO",
        "pipecat.audio": "INFO",
        "pipecat": "INFO",
    },
)

# Per-call log files land here (logs/call_<session_id>.log), one file per call.
os.makedirs("logs", exist_ok=True)

# =========================
# MULTILINGUAL CONFIG
# LANGUAGE in .env: telugu | hindi | tamil | kannada | english | auto
# =========================
_LANG_TO_PIPECAT = {
    "telugu":  Language.TE,
    "hindi":   Language.HI,
    "tamil":   Language.TA,
    "kannada": Language.KN,
    "english": Language.EN,
}

_ALL_LANGS = [Language.TE, Language.HI, Language.TA, Language.KN, Language.EN]

# Prefix used to find-and-remove a previous turn's confirmed-location
# reminder before adding this turn's — see on_user_turn_started below.
_LOCATION_REMINDER_PREFIX = "(Reminder: this caller's confirmed location is"

# Every caller is in India; the process itself isn't guaranteed to be —
# the same bot code now also runs deployable on a UTC VM (see
# bot_processors/pricing's own Azure deployment). Computing "today" with a
# fixed IST offset instead of naive local time means get_price's date
# argument (system_prompt.txt / price_lookup.py) resolves the same
# "today" a caller means regardless of the server's own timezone.
_IST = timezone(timedelta(hours=5, minutes=30))

LANGUAGE = os.getenv("LANGUAGE", "auto").lower()

# System prompt loaded from plain-text file for easy editing (no JSON escaping needed)
_PROMPT_TXT = os.path.join(os.path.dirname(__file__), "system_prompt.txt")
with open(_PROMPT_TXT, encoding="utf-8") as _f:
    SYSTEM_PROMPT = _f.read().strip()

SEED_GREETING = os.getenv("SEED_GREETING", "Hello")
BOT_GREETING  = os.getenv("BOT_GREETING",  "Hello! Welcome to Farm Vaidya AI. I can assist you with agriculture and farming.")

# For a known returning caller (name on file), the opening line is
# personalized instead of the generic BOT_GREETING — see
# _greet_and_inject_memory(). "template" builds it instantly with no extra
# latency (caller_memory.build_template_greeting); "llm" asks the model to
# write a natural one referencing their past topic, at the cost of an extra
# LLM round-trip before the greeting can be spoken (caller_summarizer.
# build_llm_greeting). Compare both, then this can collapse to one.
RETURNING_CALLER_GREETING_MODE = os.getenv("RETURNING_CALLER_GREETING_MODE", "template").strip().lower()
VOICE_NAME    = os.getenv("VOICE_NAME", "Ramya")

# Spoken once when the pipeline has been idle (no one speaking) for
# PIPELINE_IDLE_TIMEOUT_SECS — gives the caller a chance to respond before the
# call is actually ended.
IDLE_WARNING_TEXT = os.getenv(
    "IDLE_WARNING_TEXT",
    "మీరు అక్కడ ఉన్నారా? మీరు బిజీగా ఉంటే, నేను కాల్ ముగించమంటారా?",
)
IDLE_WARNING_GRACE_SECS = float(os.getenv("IDLE_WARNING_GRACE_SECS", "12"))

# Upper bound on _run_finalize's summary/DB-write gather (see its own
# comment) — normal completion is ~1s (confirmed live), so this is a wide
# margin for a slow-but-working network, not a tight budget.
FINALIZE_TIMEOUT_SECS = float(os.getenv("FINALIZE_TIMEOUT_SECS", "20"))

# Spoken immediately when a user turn is force-stopped with no transcript at
# all (STT never returned anything for that turn — e.g. the caller's speech
# raced Soniox's connect handshake right at call start, or a genuine mid-call
# dropout). Without this, the caller just hears silence after speaking and
# hangs up, since the empty transcript is dropped before it ever reaches the
# LLM — confirmed live 2026-07-29 (call_919154315557_fbf271d5.log): caller
# spoke, got nothing back, waited ~9s, hung up. Re-prompting here is much
# faster than waiting for the 300s pipeline idle-timeout (IDLE_WARNING_TEXT
# above), which is meant for mid-call silence, not a failed turn.
EMPTY_TURN_REPROMPT_TEXT = os.getenv(
    "EMPTY_TURN_REPROMPT_TEXT",
    "క్షమించండి, మీరు చెప్పింది వినిపించలేదు. మళ్ళీ చెప్పగలరా?",
)

# Voice ID: CARTESIA_VOICE_ID (primary) → VOICE_ID (legacy fallback)
VOICE_ID = os.getenv("CARTESIA_VOICE_ID") or os.getenv("VOICE_ID", "")

# Provider selectors — read from .env, validated by each factory
STT_PROVIDER = os.getenv("STT_PROVIDER", "soniox").lower()
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "cartesia").lower()
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "vertex").lower()

# TTS initial language: mapped from LANGUAGE setting; falls back to TE for auto mode
TTS_INITIAL_LANG = _LANG_TO_PIPECAT.get(LANGUAGE, Language.TE)


async def _warmup_model_clients():
    """
    Force the same cold-start costs run_bot() pays per call — Vertex AI's
    OAuth token handshake (create_llm()), Soniox's WebSocket handshake
    (stt._connect_websocket()), Bakbak's greeting synthesis, and Silero VAD's
    onnxruntime session construction — once here at server boot instead of on
    whichever real caller happens to connect first.
    """
    try:
        await asyncio.to_thread(create_llm)
        logger.info("🔥 Vertex AI LLM client warmed (OAuth handshake done)")
    except Exception:
        logger.opt(exception=True).warning("Model warmup: LLM client failed to warm")

    try:
        stt = await asyncio.to_thread(create_stt, language_hints=_ALL_LANGS)
        if isinstance(stt, SonioxSTTService):
            stt._sample_rate = 16000
            await stt._connect_websocket()
            await stt._disconnect_websocket()
            logger.info("🔥 Soniox STT connection warmed")
    except Exception:
        logger.opt(exception=True).warning("Model warmup: STT client failed to warm")

    try:
        tts = await asyncio.to_thread(create_tts, voice_id=VOICE_ID, language=TTS_INITIAL_LANG)
        if isinstance(tts, BakbakTTSService):
            tts._sample_rate = 24000  # matches pipecat's audio_out_sample_rate default (StartFrame hasn't happened yet)
            await tts.warm_static_text(BOT_GREETING)
            await tts._close_session()
    except Exception:
        logger.opt(exception=True).warning("Model warmup: TTS greeting failed to warm")

    try:
        # Throwaway VAD instance — pays the one-time onnxruntime InferenceSession
        # build cost (imports, shared-lib loading, JIT/OS-cache warming) here so
        # every real call's own SileroVADAnalyzer() construction is cheaper.
        # Safe to discard: per-call VAD recurrent state lives in the instance
        # constructed fresh in run_bot(), never shared with this one.
        SileroVADAnalyzer(params=VADParams(stop_secs=float(os.getenv("VAD_STOP_SECS", "0.2"))))
        logger.info("🔥 Silero VAD model warmed")
    except Exception:
        logger.opt(exception=True).warning("Model warmup: VAD model failed to warm")


# =========================
# WEBSOCKET SERVER
# =========================
async def main():
    PORT = int(os.getenv("PORT", 7860))
    TATA_WEBHOOK_SECRET = os.getenv("TATA_WEBHOOK_SECRET", "")

    # ── Startup banner ────────────────────────────────────────────────────────
    # ── Resolve display names from env (always reflects current .env) ───────────
    _stt_model = {
        "soniox": os.getenv("SONIOX_MODEL", "stt-rt-v5"),
        "sarvam": os.getenv("SARVAM_STT_MODEL", "saaras:v3"),
        "gnani":  os.getenv("GNANI_STT_MODEL", "prisma_v2.5"),
    }.get(STT_PROVIDER, STT_PROVIDER)

    _llm_model = {
        "vertex": os.getenv("VERTEX_MODEL", "gemini-2.5-flash"),
        "openai": os.getenv("OPENAI_MODEL", "gpt-4o"),
        "groq":   os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
    }.get(LLM_PROVIDER, LLM_PROVIDER)

    _bakbak_name = os.getenv("BAKBAK_TTS_VOICE_NAME", "Raya")
    _bakbak_id   = os.getenv("BAKBAK_TTS_VOICE_ID", "d6a002d0-230c-49b1-a137-b8a7d564b1ae")
    _voice_display = {
        "sarvam":   f"{os.getenv('SARVAM_TTS_VOICE', 'shubh')}  model={os.getenv('SARVAM_TTS_MODEL', 'bulbul:v3')}",
        "cartesia": f"{VOICE_NAME} ({VOICE_ID or 'from env'})  model={os.getenv('CARTESIA_MODEL', 'sonic-3.5')}",
        "xai":      f"{os.getenv('XAI_TTS_VOICE', 'cove')}  model={os.getenv('XAI_TTS_MODEL', 'default')}",
        "grok":     f"{os.getenv('XAI_TTS_VOICE', 'cove')}  model={os.getenv('XAI_TTS_MODEL', 'default')}",
        "bakbak":   f"{_bakbak_name} ({_bakbak_id})  model={os.getenv('BAKBAK_TTS_MODEL', 'standard')}",
        "soniox":   f"{os.getenv('SONIOX_TTS_VOICE', 'Adrian')}  model={os.getenv('SONIOX_TTS_MODEL', 'tts-rt-v1')}",
    }.get(TTS_PROVIDER, VOICE_NAME)

    logger.info("=" * 60)
    logger.info(f"  Pipecat Agent — FARM VIADYA ({LANGUAGE.upper()})")
    logger.info(f"  Prompt     : system_prompt.txt {'✅ loaded' if os.path.exists(_PROMPT_TXT) else '❌ NOT FOUND'}")
    logger.info(f"  Voice      : {_voice_display}")
    _stt_lang_display = os.getenv("GNANI_STT_LANGUAGE", LANGUAGE) if STT_PROVIDER == "gnani" else LANGUAGE
    logger.info(f"  STT        : {STT_PROVIDER.upper()}  model={_stt_model}  lang={_stt_lang_display}")
    logger.info(f"  LLM        : {LLM_PROVIDER.upper()}  model={_llm_model}")
    _tts_lang_display = "auto (switches per user language)" if LANGUAGE == "auto" else TTS_INITIAL_LANG
    logger.info(f"  TTS        : {TTS_PROVIDER.upper()} [{_tts_lang_display}]")
    logger.info(f"  Port       : {PORT}")
    logger.info("=" * 60)
    logger.info("  STT mode   : SENTENCE-LEVEL (full sentence per turn)")
    logger.info("  Latency    : STT ~200-400ms | LLM varies | TTS ~300ms")
    logger.info("=" * 60)

    # ── Provider key validation hints (actual validation happens in factories) ─
    _stt_key_env = {
        "soniox": "SONIOX_API_KEY",
        "sarvam": "SARVAM_STT_API_KEY",
        "gnani":  "GNANI_API_KEY",
    }.get(STT_PROVIDER, "")
    _tts_key_env = {
        "cartesia": "CARTESIA_API_KEY", "sarvam": "SARVAM_TTS_API_KEY",
        "xai": "XAI_API_KEY", "grok": "XAI_API_KEY",
        "bakbak": "BAKBAK_API_KEY", "soniox": "SONIOX_API_KEY",
    }.get(TTS_PROVIDER, "")
    if _stt_key_env:
        logger.info(f"STT key     : {'✅ set' if os.getenv(_stt_key_env) else '❌ MISSING'}  ({_stt_key_env})")
    if _tts_key_env:
        logger.info(f"TTS key     : {'✅ set' if os.getenv(_tts_key_env) else '❌ MISSING'}  ({_tts_key_env})")
    if LLM_PROVIDER == "vertex":
        logger.info(f"GCP PROJECT : {os.getenv('GOOGLE_CLOUD_PROJECT', 'NOT SET')}")
        logger.info(f"service-account.json: {'✅ found' if os.path.exists('service-account.json') else '❌ NOT FOUND'}")
    elif LLM_PROVIDER == "openai":
        logger.info(f"OPENAI key  : {'✅ set' if os.getenv('OPENAI_API_KEY') else '❌ MISSING'}")
    elif LLM_PROVIDER == "groq":
        logger.info(f"GROQ key    : {'✅ set' if os.getenv('GROQ_API_KEY') else '❌ MISSING'}")
    logger.info("=" * 60)

    await init_pool()
    await init_schema()
    logger.info("✅ PostgreSQL pool + schema ready")

    # ── RAG warmup (kicked off now, awaited just before we go live) ─────────────
    # Loads the Vertex embedding model + connects to Qdrant off the event loop,
    # so the first live call's RAG lookup doesn't pay this cold-start cost
    # (previously several seconds, blocking the whole pipeline mid-call).
    logger.info("🔥 Warming up RAG (embedding model + Qdrant)…")
    rag_warmup_task = asyncio.create_task(asyncio.to_thread(rag_warmup))

    # ── STT/LLM connection warmup (see _warmup_model_clients docstring) ────────
    # Pays the Vertex OAuth + Soniox WebSocket handshake cost once at boot
    # instead of on the first real caller after every restart.
    logger.info("🔥 Warming up Vertex AI + Soniox connections…")
    model_warmup_task = asyncio.create_task(_warmup_model_clients())

    app = FastAPI()

    @app.post("/webhook/tata/call-status")
    async def tata_call_status(request: Request, x_webhook_secret: str = Header(default="")):
        if TATA_WEBHOOK_SECRET and x_webhook_secret != TATA_WEBHOOK_SECRET:
            logger.warning("🚫 Tata webhook: bad/missing X-Webhook-Secret header")
            return {"status": "unauthorized"}

        payload = await request.json()
        logger.info(f"📞 Tata call-status webhook: {payload}")
        return {"status": "ok"}

    @app.post("/outbound/call_support")
    async def outbound_call_support(request: Request):
        body = await request.json()
        customer_number = body.get("customer_number", "")
        api_key = body.get("api_key", "")
        caller_id = body.get("caller_id", "")
        result = await trigger_customer_first_call(
            customer_number=customer_number,
            api_key=api_key,
            caller_id=caller_id,
        )
        return result

    @app.websocket("/ws/smartflo")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()

        # session_id tags every log line for this call (console + its own
        # file under logs/), distinguishing concurrent calls from each other.
        # Set via ContextVar (not logger.bind()) so it propagates into
        # asyncio.to_thread()/create_task() child contexts too — this reaches
        # provider factories (providers/*.py) and pipecat's own internals,
        # not just code lexically inside run_bot().
        session_id = uuid.uuid4().hex[:8]
        _current_session_id.set(session_id)
        _display_id_by_session[session_id] = session_id
        sink_id = logger.add(
            f"logs/call_{session_id}.log",
            level="DEBUG",
            format=_LOG_FORMAT,
            filter=lambda record, sid=session_id: record["extra"].get("session_id") == sid,
        )
        # Mutable so _rename_call_log() (called once the caller's number is
        # known, from deep inside run_bot()) can swap in the renamed file's
        # sink id and have it seen here too.
        sink_ref = [sink_id]

        try:
            await run_bot(websocket, session_id, sink_ref)
        except WebSocketDisconnect:
            logger.info("📵 WebSocket disconnected")
        except Exception:
            logger.exception("❌ Unhandled error in call")
        finally:
            logger.remove(sink_ref[0])
            _display_id_by_session.pop(session_id, None)

    await rag_warmup_task
    logger.info("✅ RAG warm")
    await model_warmup_task
    logger.info("✅ Vertex AI + Soniox warm")

    logger.info(f"✅ Agent LIVE → ws://localhost:{PORT}/ws/smartflo")
    logger.info("👉 Run:  ngrok http 7860")
    logger.info("👉 Smartflow URL:  wss://<ngrok-id>.ngrok-free.dev/ws/smartflo")

    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=PORT,
        log_level="warning",
        ws_ping_interval=20,
        ws_ping_timeout=20,
    )
    server = uvicorn.Server(config)
    try:
        await server.serve()
    finally:
        await close_pool()


def _rename_call_log(session_id: str, old_sink_id: int, caller_number: str) -> int:
    """Once the caller's number is known, switches this call's log file from
    logs/call_<session_id>.log to logs/call_<caller_number>_<session_id>.log
    so it's identifiable by number at a glance, not just by a random id.
    loguru holds the file open, so this means removing the old sink,
    renaming the file on disk, then re-adding a sink at the new path with the
    same routing filter (still keyed on session_id — see _inject_session_id).
    Best-effort: falls back to re-opening the original path if anything
    fails, so a rename hiccup never breaks call logging."""
    old_path = f"logs/call_{session_id}.log"
    new_path = f"logs/call_{caller_number}_{session_id}.log"
    logger.remove(old_sink_id)
    try:
        if os.path.exists(old_path):
            os.rename(old_path, new_path)
        target_path = new_path
    except OSError:
        logger.opt(exception=True).warning(f"⚠️ failed to rename {old_path} → {new_path}")
        target_path = old_path
    return logger.add(
        target_path,
        level="DEBUG",
        format=_LOG_FORMAT,
        filter=lambda record, sid=session_id: record["extra"].get("session_id") == sid,
    )


async def run_bot(websocket: WebSocket, session_id: str, sink_ref: list):
    """Build and run one fully isolated pipeline for a single Smartflow call.

    Called fresh for every WebSocket connection — every processor, the
    context, and the Pipeline/PipelineTask below are local to this call and
    are garbage-collected when it ends, so concurrent calls can never share
    or leak state into each other (unlike the old single-shared-pipeline
    design this replaced).

    Session-based logging is handled via the _current_session_id ContextVar
    (set in websocket_endpoint) + a loguru patcher, not by passing a logger
    object around — every plain `logger.info(...)` call anywhere in this
    call's execution, including in provider factories and pipecat internals,
    is tagged automatically.
    """
    # ── Transport ────────────────────────────────────────────────────────────
    serializer = SmartflowFrameSerializer()
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            serializer=serializer,
        ),
    )

    # ── VAD : Silero (detects silence locally, fires finalize to STT) ────────
    vad_stop_secs = float(os.getenv("VAD_STOP_SECS", "0.2"))
    vad_audio_idle_timeout = float(os.getenv("VAD_AUDIO_IDLE_TIMEOUT", "1.0"))
    vad = VADProcessor(
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=vad_stop_secs)),
        audio_idle_timeout=vad_audio_idle_timeout,
    )

    # ── STT / LLM / TTS — selected by *_PROVIDER in .env ────────────────────
    # create_stt() itself is fast (no network — just SDK/client init). The
    # Soniox websocket handshake used to be manually pre-connected here in the
    # background, but that raced against the pipeline's own StartFrame → STT
    # _connect() (both would try to connect at once whenever pipeline
    # construction finished before the pre-connect did — see stt_factory.py's
    # _FastSonioxSTTService for why StartFrame's own connect no longer blocks
    # the greeting, which makes this manual pre-connect unnecessary).
    stt = await asyncio.to_thread(create_stt, language_hints=_ALL_LANGS)

    llm, tts = await asyncio.gather(
        asyncio.to_thread(create_llm),
        asyncio.to_thread(create_tts, voice_id=VOICE_ID, language=TTS_INITIAL_LANG),
    )

    # Temporary timing checkpoints to pinpoint the Pipeline/PipelineTask/
    # PipelineRunner construction cost — remove once the source is confirmed.
    _ctor_t0 = time.monotonic()

    # ── Shared latency state + two logger instances ──────────────────────────
    # latency_early: placed right after STT — sees TranscriptionFrame /
    #                InterimTranscriptionFrame BEFORE context_aggregator consumes them
    # latency_late : placed after TTS — sees TTSAudioRawFrame
    # cost_tracker : placed between LLM and TTS — sees LLMTextFrame (consumed
    #                by TTS and never reaches latency_late), records first-token time
    _lat_state    = _LatencyState()
    latency_early = LatencyLogger(state=_lat_state)
    # Only the late instance logs fact_performance rows — see
    # LatencyLogger's TTS branch, the point where a full turn's STT/LLM/TTS
    # timings are all available together.
    latency_late  = LatencyLogger(state=_lat_state, serializer=serializer)
    echo_buf      = EchoBuffer(ttl_seconds=8.0)
    echo_filter   = EchoFilter(echo_buffer=echo_buf)
    transcript_corrector = TranscriptCorrector()
    cost_tracker  = CallCostTracker(state=_lat_state, echo_buffer=echo_buf, serializer=serializer)

    # ── Multilingual TTS switcher (auto mode only) ───────────────────────────
    lang_switcher = MultilingualTTSSwitcher(tts) if LANGUAGE == "auto" else None

    # Number-word conversion is Telugu-only for now (see TTSTextSanitizer) —
    # this agent is multilingual (LANGUAGE=auto switches TTS between Telugu/
    # Hindi/Tamil/Kannada/English per detected turn), so it must know the
    # *current* call language, not assume Telugu always.
    tts_sanitizer = TTSTextSanitizer(lang_switcher=lang_switcher, default_language=TTS_INITIAL_LANG)

    # ── Context ──────────────────────────────────────────────────────────────
    # Computed fresh per call (not once at process start, since this process
    # can run for days) — needed for get_price's optional date argument:
    # the LLM has no other way to turn "the day before yesterday" or "the
    # 9th" into a real DD-MM-YYYY date to pass it (see price_lookup.py's
    # get_price docstring).
    todays_date_line = f"Today's date is {datetime.now(_IST).strftime('%d-%m-%Y')} (DD-MM-YYYY)."
    messages = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n\n{todays_date_line}"},
        {"role": "user",   "content": SEED_GREETING},
        # BOT_GREETING removed here — context_aggregator.assistant() captures it
        # automatically when TTSSpeakFrame is spoken, so adding it here caused a
        # duplicate model turn that confused the LLM.
    ]
    # context is built with no tools yet, then set_tools() right after —
    # save_caller_location needs a reference to inject a system message
    # straight into context the instant a location save succeeds (see
    # make_save_caller_location's context arg below), which means context
    # has to exist before that closure is built, not after.
    context = LLMContext(messages, tools=[])

    # Populated either by _greet_and_inject_memory (returning caller, from
    # get_location()) or by save_caller_location's success path (new caller,
    # mid-call) — read on_user_turn_started, below, to re-inject a short
    # location reminder into context on EVERY turn once known, not just
    # once. A single one-time injection right after the location becomes
    # known was tried first and wasn't reliable enough on its own: confirmed
    # live (call_919390427476_03c11c0d.log, 2026-08-07) the LLM still asked
    # "which district?" for a price question just ONE turn after that exact
    # system message was injected, plainly visible right above it in
    # context. Re-stating it fresh on every subsequent turn — maximally
    # recent, right next to whatever the caller just said — is the blunt
    # but much more reliable fix; the extra prompt tokens this costs are
    # negligible next to what a single call's context already runs (see
    # bot_processors/latency.py's per-call LLM cost logging).
    # Declared before the tool closures below since save_caller_location
    # needs a reference to populate it.
    _confirmed_location: dict = {}

    # get_weather/get_price/get_price_all_markets/confirm_location/
    # save_caller_location/save_caller_name/lookup_place_by_pincode/
    # collect_pincode_digits are real LLM tool calls (see
    # bot_processors/weather_lookup.py, bot_processors/price_lookup.py,
    # bot_processors/location_lookup.py) — each closed over this call's
    # serializer purely so tool-call outcomes can still be logged with the
    # right call_id; the LLM only ever sees get_weather(location: str) /
    # get_price(commodity, state, district) / get_price_all_markets(
    # commodity, state) / confirm_location(pincode, place, state) /
    # save_caller_location(state, district, pincode, mandal, village,
    # market) / save_caller_name(name) / lookup_place_by_pincode(pincode) /
    # collect_pincode_digits(digits, reset). collect_pincode_digits is also
    # what gives lookup_place_by_pincode/confirm_location a reliably-complete
    # 6-digit pincode in the first place — see its docstring in
    # location_lookup.py for why the digit-counting moved out of the prompt.
    get_weather = make_get_weather(serializer)
    get_price = make_get_price(serializer)
    get_price_all_markets = make_get_price_all_markets(serializer)
    confirm_location = make_confirm_location(serializer)
    save_caller_location = make_save_caller_location(serializer, context, _confirmed_location)
    save_caller_name = make_save_caller_name(serializer)
    lookup_place_by_pincode = make_lookup_place_by_pincode(serializer)
    collect_pincode_digits = make_collect_pincode_digits(serializer)
    context.set_tools([
        get_weather, get_price, get_price_all_markets, confirm_location,
        save_caller_location, save_caller_name, lookup_place_by_pincode,
        collect_pincode_digits,
    ])

    # Pending "are you there?" grace-period task, set while we're waiting to
    # see if the caller responds to the idle warning (see on_idle_timeout
    # below). Cancelled the moment the caller starts speaking again.
    _idle_warning_task: asyncio.Task | None = None

    # Monotonic timestamp of the current user turn's start, used by
    # on_user_turn_stopped to decide whether _lat_state.last_final_text was
    # captured during THIS turn (safe to recover) vs. a stale leftover from
    # an earlier, already-handled turn.
    _turn_start_monotonic: float = 0.0

    # Set once in _greet_and_inject_memory, read in on_client_disconnected to
    # compute duration_seconds for the calls table row.
    _call_start_dt: datetime | None = None

    # Shared in-flight task for _finalize_call — it's reachable from three
    # independent call-ending paths (client disconnect, idle timeout,
    # escalation), and on an escalated call the transfer causes the transport
    # to disconnect almost immediately, so on_client_disconnected can fire
    # while escalation's own _finalize_call is still awaiting the LLM
    # summarization. A plain "already ran" boolean would let the second
    # caller skip past and call task.cancel() while that summarization is
    # still in flight, killing it — storing the Task instead means the
    # second caller awaits the same run rather than skipping it.
    _finalize_task: "asyncio.Task | None" = None

    # ── Turn detection ───────────────────────────────────────────────────────
    # Turn-end is decided by VAD silence + a flat post-VAD timeout — no ML
    # turn-completion gate. Previously this used
    # TurnAnalyzerUserTurnStopStrategy(LocalSmartTurnAnalyzerV3), which added
    # up to SMART_TURN_STOP_SECS (1.2s) of stall on the ~25% of turns it
    # wasn't confident about (measured in session logs). Removed 2026-07-09
    # to cut that stall; trade-off is more reliance on a short fixed pause
    # after VAD stop, which is more prone to cutting off slow/hesitant
    # speakers mid-sentence. Needs live testing to confirm this doesn't
    # regress cutoffs before treating it as final.
    #
    # SpeechTimeoutUserTurnStopStrategy still keeps an STT-latency safety
    # net (ttfs_p99_latency, from SONIOX_TTFS_P99_LATENCY) so a turn isn't
    # finalized before STT could plausibly still be sending text.
    #
    # user_speech_timeout bumped 0.05 -> 0.3 on 2026-07-24: confirmed live
    # (call_917382700894_36b029f4.log, call_917382700894_3d0ec1e2.log) that
    # 50ms was cutting turns off mid-sentence on ordinary thinking-pauses and
    # filler words ("ఆ...", "అంటే"), not just genuine turn ends. 300ms is a
    # normal pause-detection threshold — still far short of the old ML
    # analyzer's up-to-1.2s stall — and gives hesitant speakers room to
    # breathe without reintroducing that latency cost. Re-tune from live
    # calls if cutoffs or response latency are still off after this change.
    #
    # start=[VADUserTurnStartStrategy()] explicitly drops the pipecat default's
    # second start strategy, TranscriptionUserTurnStartStrategy. That one fires
    # a new "user turn started" (and broadcasts an interruption, killing
    # in-flight TTS) on ANY InterimTranscriptionFrame/TranscriptionFrame, with
    # no check that it's actually new speech. Soniox's stream keeps emitting
    # frames for a while after finalize() is called, so it was repeatedly
    # false-triggering turn-starts seconds after the real turn had already
    # stopped — cutting the bot's reply off mid-syllable and cascading into
    # phantom follow-up turns from STT noise (confirmed in
    # logs/call_919390427476_3cf7e6fa.log, 2026-07-23). VAD alone already
    # catches real barge-in correctly, so it's kept as the sole start trigger.
    turn_strategies = UserTurnStrategies(
        start=[VADUserTurnStartStrategy()],
        stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.3)],
    )
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(user_turn_strategies=turn_strategies),
    )
    rag_injector = RAGInjector(context, serializer=serializer)
    context_trimmer = ContextTrimmer(context)
    escalation_detector = EscalationDetector(serializer)
    call_end_detector = CallEndDetector()

    logger.debug(f"⏱️  processor construction: {(time.monotonic() - _ctor_t0) * 1000:.0f}ms")

    # ── Turn detection visibility ───────────────────────────────────────────
    # Its internal logging is DEBUG-level and gets swallowed by the
    # "pipecat": "INFO" catch-all filter above — these handlers surface it.
    user_aggregator = context_aggregator.user()

    @user_aggregator.event_handler("on_user_turn_started")
    async def on_user_turn_started(_aggregator, strategy):
        nonlocal _idle_warning_task, _turn_start_monotonic
        logger.info(f"🗣️  Turn START  | start_strategy={type(strategy).__name__}")
        _turn_start_monotonic = time.monotonic()
        # Reset RAG's turn buffer here rather than on the raw
        # VADUserStartedSpeakingFrame — that frame fires on VAD's own
        # start_secs/stop_secs hangover timing and can flicker mid-utterance
        # on an ordinary pause, wiping the buffer before the turn is actually
        # over (confirmed live 2026-07-24, call_917330771348_2f83e04b.log).
        # on_user_turn_started is debounced to fire once per logical turn.
        rag_injector.reset_turn()
        if _idle_warning_task is not None:
            logger.info("✅ Caller responded to idle warning — call continues")
            _idle_warning_task.cancel()
            _idle_warning_task = None
        if _confirmed_location:
            loc = _confirmed_location
            village_bit = f", village {loc['village']}" if loc.get("village") else ""
            # Drop any reminder injected for a previous turn before adding
            # this turn's — added fresh every turn once location is
            # confirmed, so without this it silently piles up one near-
            # identical copy per turn for the rest of the call. Same fix
            # RAGInjector already needed for its own passage injection (see
            # bot_processors/rag/rag.py) — role="system" collapses to a
            # fake "user" turn for Gemini/Vertex (it has no inline system
            # role — see gemini_adapter.py's _from_standard_message), and
            # several near-identical stacked fake user turns is exactly
            # what led the model to echo a literal fragment of this
            # reminder back as spoken output instead of the real answer
            # (confirmed live 2026-08-11, call_919949070894_3032625a.log —
            # TTS spoke "Use it directly for get_price/get_price_all_markets/
            # get_weather whenever they" verbatim to the caller).
            context._messages[:] = [
                m for m in context._messages
                if not (
                    m.get("role") == "system"
                    and isinstance(m.get("content"), str)
                    and m["content"].startswith(_LOCATION_REMINDER_PREFIX)
                )
            ]
            context.add_message({
                "role": "system",
                "content": (
                    f"{_LOCATION_REMINDER_PREFIX} {loc['district']}, "
                    f"{loc['state']}{village_bit} — already known, never ask for district/"
                    "state/village/pincode again. Use it directly for get_price/"
                    "get_price_all_markets/get_weather whenever they don't name a "
                    "different place this turn.)"
                ),
            })

    @user_aggregator.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(_aggregator, strategy, message):
        strategy_name = type(strategy).__name__ if strategy else "timeout"
        logger.info(f"🛑 Turn STOP   | stop_strategy={strategy_name} | text={message.content!r}")
        if strategy is None and not message.content.strip():
            # The aggregator's own buffer came back empty, but that doesn't
            # necessarily mean the caller was silent — confirmed live
            # (call_919390427476_0ccb2fd9.log, 2026-08-05) that a real,
            # correctly-transcribed answer can still get wiped from the
            # aggregator's `_full_user_turn_aggregation` before this forced
            # timeout flushes it (VAD restarting mid-utterance resets that
            # buffer — see the on_user_turn_started comment above). Recover
            # from _lat_state's independent copy of the last final transcript
            # when it was captured during this same turn and hasn't already
            # been used for a reply.
            if (
                _lat_state.last_final_text
                and not _lat_state.last_final_consumed
                and _lat_state.last_final_ts >= _turn_start_monotonic
            ):
                recovered_text = _lat_state.last_final_text
                _lat_state.last_final_consumed = True
                logger.info(
                    f"♻️  Recovered lost turn text from latency tracker: {recovered_text!r}"
                )
                await task.queue_frame(
                    LLMMessagesAppendFrame(
                        messages=[{"role": "user", "content": recovered_text}],
                        run_llm=True,
                    )
                )
            else:
                logger.info("🔁 Empty forced turn-stop — re-prompting caller instead of leaving dead air")
                await task.queue_frame(TTSSpeakFrame(text=EMPTY_TURN_REPROMPT_TEXT))

    @user_aggregator.event_handler("on_user_turn_stop_timeout")
    async def on_user_turn_stop_timeout(_aggregator):
        logger.warning("⏱️  Turn STOP TIMEOUT — no stop strategy fired in time, forced stop")

    # ── Pipeline ─────────────────────────────────────────────────────────────
    _pipeline_t0 = time.monotonic()
    pipeline = Pipeline([
        transport.input(),

        # Voice Activity Detection
        vad,

        # Speech-to-Text
        stt,

        # Dictionary correction — fuzzy/phonetic safety net for whatever
        # Soniox's native context biasing doesn't fully catch
        transcript_corrector,

        # Logging & Filtering
        latency_early,
        echo_filter,

        # Optional language switcher (auto mode only)
        *([lang_switcher] if lang_switcher else []),

        # RAG retrieval — injects knowledge base context before the LLM sees the turn
        rag_injector,

        # Mandi price and weather are no longer pipeline stages — get_price
        # and get_weather are real LLM tool calls registered on the context
        # itself (see get_weather/get_price = above, bot_processors/
        # weather_lookup.py, bot_processors/price_lookup.py), so the LLM
        # invokes them directly instead of a FrameProcessor pre-injecting
        # context.

        # Conversation Context
        context_aggregator.user(),

        # LLM
        llm,

        # Human escalation — detects [TRANSFER_TO_AGENT] in LLM output and
        # ends the call for a human transfer (see ESCALATION_* in .env)
        escalation_detector,

        # Caller-requested end of call — detects [END_CALL] in LLM output
        # and actually ends the call once the farewell has played (see
        # bot_processors/call_ender.py)
        call_end_detector,

        # Cost Tracking
        cost_tracker,

        # Strip markdown / convert digits to spoken Telugu words before TTS
        tts_sanitizer,

        # Text-to-Speech
        tts,

        # TTS Latency Tracking
        latency_late,

        # Output
        transport.output(),

        # Assistant Context
        context_aggregator.assistant(),

        # Caps how many prior turns get resent to the LLM each request, so a
        # long call's context (and LLM first-token latency) doesn't balloon.
        context_trimmer,
    ])
    logger.debug(f"⏱️  Pipeline([...]) construction: {(time.monotonic() - _pipeline_t0) * 1000:.0f}ms")

    _task_t0 = time.monotonic()
    pipeline_idle_timeout_secs = float(os.getenv("PIPELINE_IDLE_TIMEOUT_SECS", "300"))
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            # No allow_interruptions field here (unlike older pipecat) --
            # confirmed by grepping the whole installed pipecat 1.7.0 source
            # tree, the string doesn't appear anywhere in the package at
            # all. Passing it used to silently no-op (PipelineParams is a
            # pydantic model that drops unknown kwargs rather than raising),
            # so removing it changes nothing behaviorally. Interruptions are
            # on by default in this version and are governed per-frame by
            # UninterruptibleFrame (see frame_processor.py's
            # _start_interruption) plus the VAD/speech-timeout turn
            # strategies already wired up below, not a global toggle here.
            enable_metrics=True,
            # Separate flag from enable_metrics above — pipecat gates
            # STTUsageMetricsData/LLMUsageMetricsData/TTSUsageMetricsData
            # behind BOTH FrameProcessor.can_generate_metrics() (tied to
            # enable_metrics) AND FrameProcessor.usage_metrics_enabled (tied
            # to THIS flag) — see pipecat/processors/frame_processor.py's
            # start_llm_usage_metrics/start_stt_usage_metrics. Without this,
            # bot_processors/latency.py's STT/LLM cost tracking (the "🎙️
            # CALL END"/"🧠 CALL END" summary lines) never fires: confirmed
            # live, a real call had enable_metrics=True the whole time and
            # still logged zero STT/LLM usage events, only TTS (which is
            # tracked from real PCM frames, not this metrics path).
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=pipeline_idle_timeout_secs,
        # Speak a warning first (see on_idle_timeout below) instead of
        # silently dropping the call the instant the idle timer fires.
        cancel_on_idle_timeout=False,
    )
    escalation_detector.set_task(task)
    call_end_detector.set_task(task)
    logger.debug(f"⏱️  PipelineTask(...) construction: {(time.monotonic() - _task_t0) * 1000:.0f}ms")

    async def _run_finalize(reason: str) -> None:
        cost_tracker.log_summary()
        end_dt = datetime.now(timezone.utc)
        duration_seconds = int((end_dt - _call_start_dt).total_seconds()) if _call_start_dt else None
        turns = [m for m in context.messages if m.get("role") in ("user", "assistant")]
        try:
            # summarize_and_save_call makes a live LLM call — return_exceptions=True
            # below only protects against one of these three raising, it does
            # NOT bound how long a hung network call can block this gather.
            # Confirmed live (call_919390427476_345979fe.log, 2026-08-07): a
            # real DNS/network outage took down Soniox and Bakbak mid-call,
            # and the same outage evidently stalled this LLM summarization
            # call too — _finalize_call() (which every caller of this,
            # including _end_call_if_still_idle's own task.cancel() right
            # after it, awaits) never returned, so the idle-timeout
            # auto-hangup — the one mechanism meant to end an unresponsive
            # call — never actually cancelled the pipeline. The "are you
            # there?" warning re-fired two more times, ~60s apart, before the
            # call finally closed only because Smartflow's own side hung up.
            # A bounded wait here guarantees _finalize_call always returns
            # and task.cancel() always eventually runs, network outage or not.
            await asyncio.wait_for(
                asyncio.gather(
                    summarize_and_save_call(llm, context, serializer.caller_number, serializer.call_id),
                    end_call(serializer.call_id, end_dt, duration_seconds, reason, _lat_state.detected_language),
                    save_conversation_messages(serializer.call_id, turns, _lat_state.turn_latencies_ms),
                    return_exceptions=True,
                ),
                timeout=FINALIZE_TIMEOUT_SECS,
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"⚠️ _run_finalize: summary/DB writes didn't complete within "
                f"{FINALIZE_TIMEOUT_SECS:.0f}s (reason={reason!r}) — giving up on them "
                "so call teardown isn't blocked; this call's memory/transcript may be incomplete"
            )

    async def _finalize_call(reason: str) -> None:
        """Summarizes+saves this call's memory, logs it to the calls table,
        and persists its transcript — awaited (not fire-and-forget) so it
        completes before whichever caller below tears the pipeline down via
        task.cancel(). Must run exactly once: on_client_disconnected, the
        idle-timeout path, and escalation.py's _escalate() are three
        independent ways a call can end, and on an escalated call the
        transfer can make on_client_disconnected fire mere milliseconds after
        escalation's own call here — so the second caller must await the
        same run rather than skip past it (a boolean "already ran" flag
        would let it fall through to task.cancel() while the first run is
        still awaiting the LLM, killing it mid-flight)."""
        nonlocal _finalize_task
        if _finalize_task is None:
            _finalize_task = asyncio.create_task(_run_finalize(reason))
        await _finalize_task

    escalation_detector.set_finalize_callback(_finalize_call)
    call_end_detector.set_finalize_callback(_finalize_call)

    async def _end_call_if_still_idle():
        try:
            await asyncio.sleep(IDLE_WARNING_GRACE_SECS)
        except asyncio.CancelledError:
            return
        logger.info("📴 No response after idle warning — ending call")
        await _finalize_call("idle timeout - no response after warning")
        await task.cancel(reason="idle timeout - no response after warning")

    @task.event_handler("on_idle_timeout")
    async def on_idle_timeout(_task):
        nonlocal _idle_warning_task
        logger.info(
            f"⏳ Pipeline idle {pipeline_idle_timeout_secs:.0f}s — "
            f"speaking warning, {IDLE_WARNING_GRACE_SECS:.0f}s to respond"
        )
        await task.queue_frame(TTSSpeakFrame(text=IDLE_WARNING_TEXT))
        _idle_warning_task = asyncio.create_task(_end_call_if_still_idle())

    async def _greet_and_inject_memory():
        """Waits for Smartflow's 'start' event (carries the caller's number) —
        with a timeout fallback so a delayed/missing 'start' event can never
        hang the greeting — then decides and speaks the opening line: the
        static BOT_GREETING for new callers/unknown numbers, or (per
        RETURNING_CALLER_GREETING_MODE) a personalized one for known
        returning callers. Also switches this call's log tag/file from the
        random session_id to the caller's number once known (see
        _rename_call_log), and injects the caller's past-call summary into
        context for later turns."""
        try:
            await asyncio.wait_for(serializer.wait_for_start(), timeout=3.0)
        except asyncio.TimeoutError:
            logger.warning("⚠️ Smartflow 'start' event didn't arrive in time — using default greeting")

        nonlocal _call_start_dt
        if serializer.caller_number:
            _display_id_by_session[session_id] = serializer.caller_number
            sink_ref[0] = _rename_call_log(session_id, sink_ref[0], serializer.caller_number)
        _call_start_dt = datetime.now(timezone.utc)
        # record_call/start_call are just bookkeeping writes — neither depends
        # on, nor is depended on by, load_latest_summary/get_location, so run
        # all four concurrently instead of paying their DB round trips one
        # after another before the (already-slow) LLM greeting call can even
        # start.
        _, _, latest, location = await asyncio.gather(
            record_call(serializer.caller_number),
            start_call(serializer.call_id, serializer.caller_number, _call_start_dt),
            load_latest_summary(serializer.caller_number),
            get_location(serializer.caller_number),
        )
        greeting_text = BOT_GREETING
        if latest and latest.get("name"):
            name = latest["name"]
            _track_task(update_contact_name(serializer.caller_number, name))
            if RETURNING_CALLER_GREETING_MODE == "llm":
                greeting_text = await build_llm_greeting(
                    llm, name, latest.get("summary", ""), latest.get("last_topic", "")
                ) or build_template_greeting(name)
            else:
                greeting_text = build_template_greeting(name)

        await task.queue_frame(TTSSpeakFrame(text=greeting_text))

        if latest:
            context.add_message({"role": "system", "content": note_from_summary(latest)})
            logger.info(
                f"🧠 Injected returning-caller context for {serializer.caller_number} "
                f"— from call_id={latest['call_id'] or 'n/a'} at {latest['timestamp']}, "
                f"name={latest['name'] or 'unknown'}\nSummary used: {latest['summary']}"
            )

        if location:
            village_note = f", village {location['village']}" if location.get("village") else ""
            mandal_note = f", mandal {location['mandal']}" if location.get("mandal") else ""
            pincode_note = f" (pincode {location['pincode']})" if location.get("pincode") else ""
            market_note = f", nearest market yard {location['market']}" if location.get("market") else ""
            context.add_message({
                "role": "system",
                "content": (
                    f"This caller's confirmed farming location is {location['district']}, "
                    f"{location['state']}{mandal_note}{village_note}{pincode_note}{market_note}. "
                    "This caller's name and location metadata are already known — do not ask for "
                    "their name, state, district, mandal, village, or pincode again this call. "
                    "Use this district/state automatically for get_price and get_weather whenever "
                    "the caller doesn't name a different place this turn."
                ),
            })
            logger.info(
                f"📍 Injected confirmed location for {serializer.caller_number}: "
                f"{location.get('market') or location['district']}, {location['state']}"
            )
            # Also feeds on_user_turn_started's per-turn reminder (see
            # _confirmed_location's own comment above, near its declaration)
            # — this one-time injection alone wasn't reliable enough even
            # for the already-known-at-call-start case.
            _confirmed_location.update({
                "district": location["district"], "state": location["state"],
                "village": location.get("village", ""),
            })

    @transport.event_handler("on_client_connected")
    async def on_client_connected(_transport, _websocket):
        _lat_state.call_active = True
        logger.info("📞 Smartflow connected!")
        _track_task(_greet_and_inject_memory())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(_transport, _websocket):
        nonlocal _idle_warning_task
        _lat_state.call_active = False
        if _idle_warning_task is not None:
            _idle_warning_task.cancel()
            _idle_warning_task = None

        logger.info("📵 Call ended — client disconnected")
        await _finalize_call("client disconnected")
        await task.cancel(reason="client disconnected")

    logger.debug(f"⏱️  total construction (post-Soniox-prewarm → pre-runner.run): {(time.monotonic() - _ctor_t0) * 1000:.0f}ms")
    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Shutting down (Ctrl+C)")
