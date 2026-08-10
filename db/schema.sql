-- voice_agent database schema for the Farm Vaidya AI voice agent (PostgreSQL).
--
-- Star-schema style: one dimension table (dim_contacts) plus fact tables
-- (fact_*) for everything that happens during/around a call. Postgres
-- enforces foreign keys natively (unlike SQLite, which needed
-- "PRAGMA foreign_keys = ON" per connection), and has a native BOOLEAN type,
-- so boolean-like columns (success) are BOOLEAN here instead of the old
-- INTEGER CHECK (0/1).

-- Caller registry — tracks which phone numbers have called before, their
-- name (if known), how many times, and (once verbally captured via the
-- metadata-capture flow — see bot_processors/location_lookup.py) their
-- state/district/mandal/village/pincode, so later calls don't need to ask
-- again. confirmed_market is the specific Agmarknet mandi the caller
-- recognized by name, when that's separately been confirmed for pricing
-- (most precise); confirmed_district is what get_price/get_weather actually
-- key off of. confirmed_mandal/confirmed_village are the caller's own plain
-- location words, spoken as part of the metadata-capture sequence
-- (name -> state -> district -> mandal -> village -> pincode) and stored
-- as-is, independent of the market-yard flow.
CREATE TABLE IF NOT EXISTS dim_contacts (
    phone_number          TEXT PRIMARY KEY,
    name                  TEXT,
    first_seen            TIMESTAMPTZ NOT NULL,
    last_seen             TIMESTAMPTZ NOT NULL,
    call_count            INTEGER NOT NULL DEFAULT 0,
    confirmed_state       TEXT,
    confirmed_district    TEXT,
    confirmed_mandal      TEXT,
    confirmed_village     TEXT,
    confirmed_market      TEXT,
    confirmed_pincode     TEXT,
    location_confirmed_at TIMESTAMPTZ
);

-- dim_contacts predates the caller-location-confirmation feature -- these
-- backfill the columns on any database created before they existed
-- (CREATE TABLE IF NOT EXISTS above is a no-op once the table already exists).
ALTER TABLE dim_contacts ADD COLUMN IF NOT EXISTS confirmed_state TEXT;
ALTER TABLE dim_contacts ADD COLUMN IF NOT EXISTS confirmed_district TEXT;
ALTER TABLE dim_contacts ADD COLUMN IF NOT EXISTS confirmed_mandal TEXT;
ALTER TABLE dim_contacts ADD COLUMN IF NOT EXISTS confirmed_village TEXT;
ALTER TABLE dim_contacts ADD COLUMN IF NOT EXISTS confirmed_market TEXT;
ALTER TABLE dim_contacts ADD COLUMN IF NOT EXISTS confirmed_pincode TEXT;
ALTER TABLE dim_contacts ADD COLUMN IF NOT EXISTS location_confirmed_at TIMESTAMPTZ;

-- One row per phone call — the parent record everything else hangs off of.
CREATE TABLE IF NOT EXISTS fact_sessions (
    call_id           TEXT PRIMARY KEY,                 -- Tata Smartflo's callSid
    phone_number      TEXT NOT NULL,                     -- caller's number, digits only
    direction         TEXT NOT NULL CHECK (direction IN ('inbound', 'outbound')),
    start_time        TIMESTAMPTZ NOT NULL,
    end_time          TIMESTAMPTZ,
    duration_seconds  INTEGER,
    language          TEXT,
    status            TEXT NOT NULL DEFAULT 'in_progress'
                          CHECK (status IN ('in_progress', 'completed', 'missed', 'failed')),
    end_reason        TEXT                                 -- e.g. "client disconnected", "idle timeout"
);

CREATE INDEX IF NOT EXISTS idx_fact_sessions_phone_number ON fact_sessions (phone_number);
CREATE INDEX IF NOT EXISTS idx_fact_sessions_start_time ON fact_sessions (start_time);

-- Every message exchanged during a call (caller, bot, and system/RAG notes).
CREATE TABLE IF NOT EXISTS fact_conversations (
    id               BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    call_id          TEXT NOT NULL REFERENCES fact_sessions (call_id) ON DELETE CASCADE,
    speaker          TEXT NOT NULL CHECK (speaker IN ('user', 'assistant', 'system')),
    message          TEXT NOT NULL,
    timestamp        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    latency_ms       INTEGER                              -- time to produce this message, if applicable
);

CREATE INDEX IF NOT EXISTS idx_fact_conversations_call_id ON fact_conversations (call_id);
CREATE INDEX IF NOT EXISTS idx_fact_conversations_timestamp ON fact_conversations (timestamp);

-- Every tool/function invocation the agent makes during a call.
CREATE TABLE IF NOT EXISTS fact_toolcalls (
    id                  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    call_id             TEXT NOT NULL REFERENCES fact_sessions (call_id) ON DELETE CASCADE,
    tool_name           TEXT NOT NULL,
    input               TEXT,                             -- JSON-encoded arguments
    output              TEXT,                             -- JSON-encoded result
    success             BOOLEAN NOT NULL DEFAULT FALSE,
    execution_time_ms   INTEGER,
    timestamp           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_fact_toolcalls_call_id ON fact_toolcalls (call_id);
CREATE INDEX IF NOT EXISTS idx_fact_toolcalls_tool_name ON fact_toolcalls (tool_name);

-- Latency/performance breakdown, one row per response turn within a call.
CREATE TABLE IF NOT EXISTS fact_performance (
    id                  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    call_id             TEXT NOT NULL REFERENCES fact_sessions (call_id) ON DELETE CASCADE,
    stt_ms              INTEGER,                          -- speech-to-text time
    llm_ttft_ms         INTEGER,                          -- LLM time-to-first-token
    llm_total_ms        INTEGER,                          -- full LLM generation time
    tts_ms              INTEGER,                          -- text-to-speech time
    total_response_ms   INTEGER,                          -- end-to-end turnaround for this response
    timestamp           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_fact_performance_call_id ON fact_performance (call_id);
CREATE INDEX IF NOT EXISTS idx_fact_performance_timestamp ON fact_performance (timestamp);

-- Cumulative per-call caller summaries (one row per call). Each row's
-- summary already merges every call before it, so "this caller's context"
-- is always just the single latest row for their phone_number, not the
-- full history — see load_latest_summary() in bot_processors/caller_memory.py.
CREATE TABLE IF NOT EXISTS fact_conversation_summary (
    id               BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    phone_number     TEXT NOT NULL,
    call_id          TEXT,
    name             TEXT,
    summary          TEXT NOT NULL,
    -- Short phrase naming what THIS specific call (not the cumulative
    -- summary) was mainly about. The summary itself is topic-reorganized/
    -- condensed each call, not chronological, so "the last thing near the
    -- end of the summary" isn't reliably the most recently discussed topic —
    -- this column is the one thing the returning-caller greeting needs and
    -- can trust for recency. See bot_processors/caller_summarizer.py.
    last_topic       TEXT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_fact_conversation_summary_phone_number ON fact_conversation_summary (phone_number, created_at DESC);

-- Mandi (market) prices scraped from Agmarknet — one row per (state,
-- district, commodity, arrival_date). Replaces bot_processors/
-- last_known_prices.json (retired): that file overwrote each key in place
-- and kept only the single latest value, so an older price was gone the
-- moment a fresher one came in. Here a new arrival_date is always a NEW
-- row — genuine history, nothing lost — and only a same-day re-scrape
-- (same arrival_date) updates that day's row instead of duplicating it.
-- crop_keyword is the normalized lookup key bot_processors/price_shared.py's
-- crop_keyword_for() derives from commodity (e.g. "Onion" -> "onion") —
-- what get_price/get_price_all_markets actually filter on; commodity is
-- the display name spoken to the caller. See bot_processors/price_lookup.py
-- for the read path (a get_price/get_price_all_markets lookup is a plain
-- "latest row per district within the last N days" query against this
-- table) and bot_processors/agmarknet_scraper.py for the write path (one
-- upsert per scraped row, each scrape cycle).
CREATE TABLE IF NOT EXISTS fact_market_prices (
    id             BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    state          TEXT NOT NULL,
    district       TEXT NOT NULL,
    market         TEXT NOT NULL,                  -- specific Agmarknet mandi, e.g. "Kurnool APMC"
    commodity      TEXT NOT NULL,                  -- display name, e.g. "Onion"
    crop_keyword   TEXT NOT NULL,                  -- normalized lookup key, e.g. "onion"
    arrival_date   DATE NOT NULL,                  -- the date this price was reported for
    modal_per_kg   NUMERIC(10, 2) NOT NULL,
    min_per_kg     NUMERIC(10, 2) NOT NULL,
    max_per_kg     NUMERIC(10, 2) NOT NULL,
    arrival_qty    NUMERIC(10, 2),                 -- NULL when the site didn't report one
    arrival_unit   TEXT,
    scraped_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (state, district, crop_keyword, arrival_date)
);

-- Covers get_price_all_markets: "every district's latest row for this
-- (state, crop) within the cutoff window" — district isn't known yet.
CREATE INDEX IF NOT EXISTS idx_fact_market_prices_state_crop
    ON fact_market_prices (state, crop_keyword, arrival_date DESC);

-- Covers get_price: "this exact district's latest row for this (state, crop)".
CREATE INDEX IF NOT EXISTS idx_fact_market_prices_state_district_crop
    ON fact_market_prices (state, district, crop_keyword, arrival_date DESC);
