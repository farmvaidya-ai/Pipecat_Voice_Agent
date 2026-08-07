"""
LLM Factory — reads LLM_PROVIDER from .env and returns the configured service.

Supported providers (set LLM_PROVIDER= in .env):
    vertex    → GoogleVertexLLMService   (default)
    openai    → OpenAILLMService
    groq      → GroqLLMService (native) or OpenAI-compatible fallback

Adding a new provider:
    1. Add its env vars to .env (e.g. NEWPROVIDER_API_KEY=, NEWPROVIDER_MODEL=)
    2. Add an elif branch in create_llm() that calls a new _create_newprovider_llm() helper
    3. Write the helper function below following the same pattern
    4. Update the error message in create_llm() to list the new provider

Vertex AI retry logic:
    Google OAuth token refresh makes an HTTPS call at startup. On restricted
    networks this sometimes fails with RemoteDisconnected. The factory retries
    up to VERTEX_INIT_RETRIES times (default 3) with VERTEX_RETRY_DELAY_SECS
    (default 3) seconds between attempts before raising.
"""

import os
import threading
import time

from loguru import logger

# Vertex AI's OAuth token handshake (creds.refresh()) is a ~1-2s HTTPS round
# trip that GoogleVertexLLMService's _get_credentials() pays on *every*
# construction, even though the token it fetches is valid for ~1 hour. Since
# run_bot() builds a fresh GoogleVertexLLMService per call, this was gating
# the greeting behind a live OAuth call every single call. We cache the
# refreshed Credentials object here (module-level, guarded by a real thread
# lock since create_llm() runs inside asyncio.to_thread) and reuse it until
# it's actually expired, so only the first call per ~hour pays the refresh.
# Sharing the Credentials object (not the LLMService/genai Client, which are
# per-pipeline FrameProcessors closed at call teardown) across concurrent
# calls is safe — it's just a bearer token read by each call's own client.
_vertex_creds_lock = threading.Lock()
_vertex_creds = None


def _get_cached_vertex_credentials(credentials_path: str):
    from google.auth.transport.requests import Request
    from google.oauth2 import service_account

    global _vertex_creds
    with _vertex_creds_lock:
        if _vertex_creds is None or not _vertex_creds.valid:
            _t0 = time.monotonic()
            creds = service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            creds.refresh(Request())
            _vertex_creds = creds
            logger.debug(
                f"[LLM Factory] Vertex credentials refreshed (cold OAuth call): "
                f"{(time.monotonic() - _t0) * 1000:.0f}ms"
            )
        return _vertex_creds


def create_llm():
    """
    Instantiate and return the LLM service selected by LLM_PROVIDER in .env.

    Returns:
        Configured LLM service instance ready for pipeline use.

    Raises:
        ValueError:  Required env var is missing.
        ImportError: Provider package is not installed.
        ValueError:  LLM_PROVIDER names an unsupported provider.
    """
    provider = os.getenv("LLM_PROVIDER", "vertex").lower().strip()
    logger.info(f"[LLM Factory] provider={provider.upper()}")

    if provider == "vertex":
        return _create_vertex()
    if provider == "openai":
        return _create_openai()
    if provider == "groq":
        return _create_groq()

    raise ValueError(
        f"[LLM Factory] Unsupported LLM_PROVIDER='{provider}'. "
        "Valid values: vertex, openai, groq"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Google Vertex AI
# ──────────────────────────────────────────────────────────────────────────────

def _create_vertex():
    from google.genai.types import HarmBlockThreshold, HarmCategory, SafetySetting
    from pipecat.services.google.vertex.llm import GoogleVertexLLMService as _VertexService

    # Explicit content-safety thresholds (pipecat-ai>=1.7.0's
    # GoogleLLMService.Settings.safety_settings) — a public phone line taking
    # unscreened calls from farmers is exactly the kind of open input surface
    # worth setting deliberately rather than leaving to Gemini's implicit
    # defaults. Left at BLOCK_MEDIUM_AND_ABOVE (Gemini's own normal default)
    # rather than tightened further: this is an agricultural pest-control
    # bot, so words like "poison", "kill", "spray" are routine domain
    # vocabulary, not something a stricter threshold should be flagging.
    _VERTEX_SAFETY_SETTINGS = [
        SafetySetting(category=cat, threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE)
        for cat in (
            HarmCategory.HARM_CATEGORY_HARASSMENT,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        )
    ]

    class _FastVertexLLMService(_VertexService):
        # Override the credentials lookup only — everything else (client
        # construction, generation, per-pipeline lifecycle) is unchanged.
        @staticmethod
        def _get_credentials(credentials, credentials_path):
            return _get_cached_vertex_credentials(credentials_path)

    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        raise ValueError("[LLM Factory] GOOGLE_CLOUD_PROJECT is not set in .env")

    location          = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    model             = os.getenv("VERTEX_MODEL", "gemini-2.5-flash")
    credentials_path  = os.path.abspath("service-account.json")
    max_retries       = int(os.getenv("VERTEX_INIT_RETRIES", "3"))
    retry_delay_secs  = float(os.getenv("VERTEX_RETRY_DELAY_SECS", "3"))

    logger.info(
        f"[LLM Factory] Vertex AI model={model} "
        f"project={project_id} location={location} "
        f"credentials={credentials_path}"
    )

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            _t0 = time.monotonic()
            service = _FastVertexLLMService(
                credentials_path=credentials_path,
                project_id=project_id,
                location=location,
                # Disables Gemini's "thinking" phase, which otherwise defaults
                # to on for run_inference() (the out-of-band summarization/
                # greeting calls in caller_summarizer.py) even though the
                # normal chat pipeline already disables it by default for
                # gemini-2.5-flash. Without this, thinking silently consumes
                # the whole max_tokens budget (700/150 tokens) before any
                # visible output is produced, so the response comes back with
                # finish_reason=MAX_TOKENS and empty content.parts, which
                # pipecat's run_inference crashes on (TypeError iterating
                # None) — caught by our own try/except, but silently failing
                # summarization/greeting generation on most calls.
                settings=_VertexService.Settings(
                    model=model,
                    thinking=_VertexService.ThinkingConfig(thinking_budget=0),
                    safety_settings=_VERTEX_SAFETY_SETTINGS,
                ),
            )
            logger.debug(
                f"[LLM Factory] _FastVertexLLMService construction "
                f"(cached-creds lookup + genai.Client() build): "
                f"{(time.monotonic() - _t0) * 1000:.0f}ms"
            )
            if attempt > 1:
                logger.info(f"[LLM Factory] Vertex AI init succeeded on attempt {attempt}")
            return service
        except Exception as exc:
            last_exc = exc
            logger.warning(
                f"[LLM Factory] Vertex AI init attempt {attempt}/{max_retries} failed: {exc}"
            )
            if attempt < max_retries:
                time.sleep(retry_delay_secs)

    raise RuntimeError(
        f"[LLM Factory] Vertex AI failed to initialise after {max_retries} attempts. "
        f"Last error: {last_exc}"
    ) from last_exc


# ──────────────────────────────────────────────────────────────────────────────
# OpenAI
# ──────────────────────────────────────────────────────────────────────────────

def _create_openai():
    from pipecat.services.openai.llm import OpenAILLMService

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("[LLM Factory] OPENAI_API_KEY is not set in .env")

    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    logger.info(f"[LLM Factory] OpenAI model={model}")

    return OpenAILLMService(
        api_key=api_key,
        model=model,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Groq
# ──────────────────────────────────────────────────────────────────────────────

def _create_groq():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("[LLM Factory] GROQ_API_KEY is not set in .env")

    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # Try native pipecat Groq service first; fall back to OpenAI-compatible endpoint.
    try:
        from pipecat.services.groq.llm import GroqLLMService
        logger.info(f"[LLM Factory] Groq (native) model={model}")
        return GroqLLMService(
            api_key=api_key,
            model=model,
        )
    except ImportError:
        logger.info(
            f"[LLM Factory] Native Groq service not found — "
            f"using OpenAI-compatible endpoint. model={model}"
        )

    from pipecat.services.openai.llm import OpenAILLMService
    return OpenAILLMService(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
        model=model,
    )
