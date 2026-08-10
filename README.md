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

- **`bot_processors/`** — call-time processors and background jobs, split
  into domain subpackages:
  - **`pricing/`** — live commodity price lookup. `price_lookup.py` /
    `agmarknet_scraper.py` scrape agmarknet.gov.in via Playwright and
    upsert into the `fact_market_prices` Postgres table (`scraper_daemon.py`,
    scheduled 3x/day); `price_shared.py`, `export_prices_xlsx.py`,
    `enrich_prices_with_geo.py`, `market_yard_geo_reference.py`,
    `nationwide_market_geo_reference.py` support that pipeline.
  - **`location/`** — `location_lookup.py` (pincode/village/market-yard
    resolution), `pincode_geo_reference.py`, `weather_lookup.py`.
  - **`calls/`** — `caller_db.py` / `caller_memory.py` / `caller_summarizer.py`
    / `call_db.py` (Postgres-backed caller history and per-call state),
    plus `call_ender.py`, `escalation.py`, `outbound_call.py`, `serializer.py`.
  - **`voice/`** — `echo.py`, `latency.py`, `tts_switcher.py`,
    `number_words.py` — echo/crosstalk filtering, latency+cost tracking,
    multilingual TTS switching, spoken-number formatting per language.
  - **`rag/`** — `rag.py`, `intent_router.py` — knowledge-base retrieval
    and intent routing.
  - **`core/`** — `db_pool.py`, `voice_agent_db.py`, `task_tracker.py`,
    `context_trimmer.py` — shared infra every other subpackage depends on.
  - **`data/`** — every reference/cache JSON and xlsx file (Agmarknet
    commodity/state lists, geocoding caches, `last_known_prices.json`).
    Not code — data only. Path is `bot_processors/paths.py`'s `DATA_DIR`,
    the one place every module computes this from, rather than each file
    working out its own relative path.
- **`providers/`** — STT/TTS/LLM factory + adapters
    (`stt_factory.py`, `tts_factory.py`, `llm_factory.py`), plus
    transcript correction and domain vocabulary boosting.
- **`chonkie_rag/`** — knowledge-base ingestion pipeline (chunking,
  embedding, Qdrant search) that feeds `bot_processors/rag/rag.py`. Active
  collection is chosen via `RAG_COLLECTION` in `.env` (APS/KVK are separate
  collections). Source documents for ingestion live in `knowledge_base/`.
- **`knowledge_base/`** — the raw `.txt` source documents (APS/KVK/Grow
  It/Salam Kisan/Nacro/Rythu Nestham) that get chunked and embedded into
  Qdrant via `python -m chonkie_rag.ingest` — not read directly by the live
  bot, only by that one-off ingestion step.
- **`db/`** — Postgres schema (`schema.sql`, `migrate_to_star_schema.sql`).
- **`scripts/`** — one-off/utility scripts: `wipe_db.py` (truncate all
  caller/call tables), `view_callers.py` (print `dim_contacts`),
  `backfill_market_prices.py` (one-time JSON → `fact_market_prices`
  migration, already run).
- **`geo_validator/`** — a separate pincode/geo-coordinate validation tool,
  independent of the live bot.
- **`docs/`** — implementation write-ups (caller registration, cost
  tracking, market-price DB migration).

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
