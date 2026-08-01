# Farm Vaidya — Multilingual Voice Agent

A Pipecat-based voice bot for farmers in Andhra Pradesh/Telangana. Answers
calls in Telugu, Hindi, Tamil, Kannada, or English (auto-detected), backed
by a knowledge base (KVK/APS advisories), live commodity price lookup, and
weather lookup.

## Entry point

```
Bot.py
```

Runs the Pipecat pipeline for a single call. STT, TTS, and LLM providers
are all selected from `.env`, not hardcoded:

```
LANGUAGE      = telugu | hindi | tamil | kannada | english | auto
STT_PROVIDER  = soniox | sarvam
TTS_PROVIDER  = cartesia | sarvam
LLM_PROVIDER  = vertex | openai | groq
```

Launch with the project's own venv (avoids picking up a system Python):

```
bot.bat
```

## Layout

- **`bot_processors/`** — call-time processors and background jobs:
  - `price_lookup.py` / `agmarknet_scraper.py` — live commodity price
    lookup via Playwright against agmarknet.gov.in, with a cache
    (`last_known_prices.json`) refreshed by `scraper_daemon.py` (scheduled
    3x/day).
  - `caller_db.py` / `caller_memory.py` / `call_db.py` — Postgres-backed
    caller history and per-call state.
  - `rag.py`, `intent_router.py`, `weather_lookup.py`,
    `number_words.py` — knowledge-base retrieval, intent routing, weather,
    and spoken-number formatting per language.
- **`providers/`** — STT/TTS/LLM factory + adapters
    (`stt_factory.py`, `tts_factory.py`, `llm_factory.py`), plus
    transcript correction and domain vocabulary boosting.
- **`chonkie_rag/`** — knowledge-base ingestion pipeline (chunking,
  embedding, Qdrant search) that feeds `rag.py`. Active collection is
  chosen via `RAG_COLLECTION` in `.env` (APS/KVK are separate collections).
- **`db/`** — Postgres schema (`schema.sql`, `migrate_to_star_schema.sql`).

## Setup

1. `venv/` contains all dependencies — no `requirements.txt` yet; install
   into the venv directly if rebuilding it from scratch.
2. Copy your own `.env` (not committed) with provider API keys, `LANGUAGE`,
   `STT_PROVIDER`/`TTS_PROVIDER`/`LLM_PROVIDER`, `RAG_COLLECTION`, and
   Postgres connection settings.
3. `service-account.json` (GCP credentials, not committed) is required for
   Vertex LLM/STT if used.

## Not committed (see `.gitignore`)

`.env`, `service-account.json`, `venv/`, `logs/`, `caller_memory/`,
`chonkie_rag/cache/`, and scraper daemon pid/log files — all contain
secrets, real call data, or regeneratable artifacts.
