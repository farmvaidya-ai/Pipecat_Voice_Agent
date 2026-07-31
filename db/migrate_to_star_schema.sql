-- One-time migration: renames the original tables to the new dim_/fact_
-- star-schema names (see db/schema.sql) and retires call_summary, a
-- write-only raw-transcript dump that duplicated fact_conversations and was
-- never read anywhere. Run this once against an existing database — fresh
-- installs get the final names straight from schema.sql and don't need it.
--
-- Renaming (rather than create-new + copy + drop) keeps all existing data,
-- PKs, and sequences intact with no data movement.

ALTER TABLE callers RENAME TO dim_contacts;
ALTER TABLE dim_contacts ADD COLUMN IF NOT EXISTS name TEXT;

ALTER TABLE calls RENAME TO fact_sessions;
ALTER TABLE conversation_messages RENAME TO fact_conversations;
ALTER TABLE tool_calls RENAME TO fact_toolcalls;
ALTER TABLE performance_metrics RENAME TO fact_performance;
ALTER TABLE caller_summaries RENAME TO fact_conversation_summary;

DROP TABLE IF EXISTS call_summary;

-- Cosmetic: rename indexes to match the new table names.
ALTER INDEX IF EXISTS idx_calls_phone_number RENAME TO idx_fact_sessions_phone_number;
ALTER INDEX IF EXISTS idx_calls_start_time RENAME TO idx_fact_sessions_start_time;
ALTER INDEX IF EXISTS idx_conversation_messages_call_id RENAME TO idx_fact_conversations_call_id;
ALTER INDEX IF EXISTS idx_conversation_messages_timestamp RENAME TO idx_fact_conversations_timestamp;
ALTER INDEX IF EXISTS idx_tool_calls_call_id RENAME TO idx_fact_toolcalls_call_id;
ALTER INDEX IF EXISTS idx_tool_calls_tool_name RENAME TO idx_fact_toolcalls_tool_name;
ALTER INDEX IF EXISTS idx_performance_metrics_call_id RENAME TO idx_fact_performance_call_id;
ALTER INDEX IF EXISTS idx_performance_metrics_timestamp RENAME TO idx_fact_performance_timestamp;
ALTER INDEX IF EXISTS idx_caller_summaries_phone_number RENAME TO idx_fact_conversation_summary_phone_number;
