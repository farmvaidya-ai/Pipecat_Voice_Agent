-- voice_agent database schema for the Farm Vaidya AI voice agent (PostgreSQL).
--
-- Star-schema style: one dimension table (dim_contacts) plus fact tables
-- (fact_*) for everything that happens during/around a call. Postgres
-- enforces foreign keys natively (unlike SQLite, which needed
-- "PRAGMA foreign_keys = ON" per connection), and has a native BOOLEAN type,
-- so boolean-like columns (success) are BOOLEAN here instead of the old
-- INTEGER CHECK (0/1).

-- Caller registry — tracks which phone numbers have called before, their
-- name (if known), and how many times.
CREATE TABLE IF NOT EXISTS dim_contacts (
    phone_number TEXT PRIMARY KEY,
    name         TEXT,
    first_seen   TIMESTAMPTZ NOT NULL,
    last_seen    TIMESTAMPTZ NOT NULL,
    call_count   INTEGER NOT NULL DEFAULT 0
);

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
