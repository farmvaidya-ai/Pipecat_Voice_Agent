"""Post-call summarization: folds a just-ended call's transcript together with
the caller's existing (cumulative) summary into one updated summary (capped
at SUMMARY_MAX_WORDS, no minimum), and saves it via caller_memory for use on
the caller's next call (see Bot.py's on_client_disconnected).

Because each saved summary already absorbs every call before it, this always
merges forward from the single latest saved summary rather than the full raw
history — call 11 only needs call 10's summary, which itself already folded
in calls 1-9.
"""

import os
import re

from loguru import logger

from pipecat.processors.aggregators.llm_context import LLMContext

from bot_processors.calls.caller_memory import load_latest_summary, save_summary

# The saved summary is capped at this many words — short calls stay short,
# long/eventful histories get trimmed instead of running away in length. Set
# in .env (RETURNING_CALLER_GREETING_MODE's section) so it can be tuned
# without a code change.
SUMMARY_MAX_WORDS = int(os.getenv("SUMMARY_TARGET_WORDS", "300"))

# Default language for callers with no per-call language passed in (keeps the
# original hardcoded-Telugu behavior as the fallback for old call sites).
DEFAULT_LANGUAGE = "Telugu"


def _summary_instruction(language: str) -> str:
    """SUMMARY_INSTRUCTION, parameterized by the IVR's currently configured
    spoken language (Bot.py's LANGUAGE env var) instead of hardcoded Telugu —
    the Name/LastTopic/Summary fields all get read aloud in later greetings,
    so they must be written in whatever language the bot is actually
    speaking, not whatever the deployment originally shipped with."""
    return (
        "You maintain a running memory of a farm-advisory caller across multiple phone "
        "calls. You will be given the caller's prior history so far (if any) and the "
        "transcript of their latest call. Produce ONE updated cumulative summary that "
        "merges both: preserve the caller's name and key ongoing details (crops, past "
        "issues, products discussed, what's resolved vs. still open), and fold in "
        "what's new from this latest call. Condense or drop repeated/resolved older "
        "details as needed — do not just append the new call onto the old one. The "
        f"summary must be no more than {SUMMARY_MAX_WORDS} words — shorter is fine "
        "if that's all there is to say, just cover everything that matters. Do not "
        "pad or invent detail just to reach a length. You are analyzing transcripts, "
        "not continuing the conversation — do not reply to the caller or write a "
        "farewell. Plain text, no markdown.\n\n"
        "Respond in exactly this format:\n"
        f"Name: <caller's name only, transliterated into {language} script — this is a "
        f"{language}-speaking IVR and the name is spoken aloud inside {language} sentences, "
        "so it must never be left in Latin/English letters, and must never carry an "
        "English gloss in parentheses after it; or unknown if never stated>\n"
        f"LastTopic: <a short phrase (5-10 words), in {language}, naming what THIS "
        "latest call specifically was mainly about — not the caller's whole "
        "history, just this one call>\n"
        f"Summary: <the updated summary, at most {SUMMARY_MAX_WORDS} words, in {language}>"
    )
# Telugu (and other Indic scripts) run several tokens per word — give enough
# headroom for a full SUMMARY_MAX_WORDS response plus the Name: line and
# formatting. 700 was the old fixed value for a much looser "~300 words"
# target; scale it with the configured cap instead of hardcoding.
SUMMARY_MAX_TOKENS = max(700, SUMMARY_MAX_WORDS * 6)

def _greeting_instruction(language: str) -> str:
    """GREETING_INSTRUCTION, parameterized by the IVR's currently configured
    spoken language instead of hardcoded Telugu. No literal script examples
    here (the old Telugu ones) — a Telugu-script example could bias the model
    toward mixing scripts when the target language is something else."""
    return (
        "You are opening a phone call with a returning farm-advisory caller. You are "
        "given their name, the specific topic of their most recent call, and a "
        f"broader summary of their past calls. Write ONE short natural spoken {language} "
        "opening line — under 38 words — that follows this exact structure, in order:\n"
        "1. Greet them by name.\n"
        "2. A short pleasantry saying you're happy to talk to them again.\n"
        "3. Explicitly say you discussed something before, in your own words, not "
        "read verbatim. Always reference the given MOST RECENT topic specifically — "
        "never a different, older topic from the broader summary, even if that older "
        "topic takes up more of the summary's length. If no most-recent topic was "
        "given, reference the broader summary's overall topic instead.\n"
        "4. Ask how you can help them today.\n"
        "All 4 parts must be present, in this order, as one flowing spoken line — "
        f"do not drop the pleasantry to save words. Write entirely in {language} "
        "(words and script), plain text, no markdown, no quotes — output only the "
        "line to be spoken aloud."
    )


GREETING_MAX_TOKENS = 170

_ROLE_LABELS = {"user": "Caller", "assistant": "Bot"}
_NAME_SUMMARY_RE = re.compile(
    r"Name:\s*(.*?)\n+LastTopic:\s*(.*?)\n+Summary:\s*(.*)", re.DOTALL | re.IGNORECASE
)
# Despite SUMMARY_INSTRUCTION asking for one script only, the LLM sometimes
# glosses the name with an English transliteration in parentheses (e.g.
# "కవిత (Kavitha)") — spoken verbatim by build_template_greeting, that comes
# out as the caller's name read out twice, once per language. Strip it as a
# second line of defense regardless of what the prompt achieves.
_PAREN_GLOSS_RE = re.compile(r"\s*\([^)]*\)\s*$")


def _flatten_transcript(turns: list[dict]) -> str:
    """Renders user/assistant turns as one plain-text transcript block
    instead of a native multi-turn conversation. Passing turns as real
    role="user"/"assistant" messages to the LLM makes it treat them as a
    live dialogue to *continue* (e.g. it would just generate the next
    farewell-style reply), rather than as text to step back and analyze —
    this flattening is what makes summarization actually summarize."""
    lines = []
    for m in turns:
        content = m.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        label = _ROLE_LABELS.get(m.get("role"), m.get("role"))
        lines.append(f"{label}: {content}")
    return "\n".join(lines)


def _parse_name_and_summary(raw: str, fallback_name: str = "") -> tuple[str, str, str]:
    """Pulls the "Name: ...\nLastTopic: ...\nSummary: ..." fields out of the
    LLM's response. Falls back to the caller's previously known name if this
    call's response didn't restate it (e.g. they didn't say their name
    again). Empty last_topic if the response didn't parse (e.g. an older-
    format response) — callers should fall back to the template greeting."""
    match = _NAME_SUMMARY_RE.match(raw.strip())
    if not match:
        return fallback_name, "", raw.strip()
    extracted_name = _PAREN_GLOSS_RE.sub("", match.group(1).strip()).strip()
    name = extracted_name if extracted_name and extracted_name.lower() != "unknown" else fallback_name
    return name, match.group(2).strip(), match.group(3).strip()


_REPEATED_RUN_RE = re.compile(r"(.{15,}?)(?:[\s.,!?।]*\1){2,}", re.DOTALL)


def _dedupe_repeated_runs(text: str) -> str:
    """Safeguard against a rare LLM degeneration failure mode seen in
    production where the summarization model gets stuck repeating the same
    sentence or phrase verbatim 3+ times in a row (sometimes trailing off
    mid-word) instead of terminating normally. Collapses any such run down
    to one occurrence, regardless of what caused it."""
    return _REPEATED_RUN_RE.sub(r"\1", text)


def _cap_word_count(summary: str, max_words: int) -> str:
    """SUMMARY_INSTRUCTION already asks the model to stay within max_words,
    but LLMs don't reliably respect a hard cap — this just trims if it ran
    over. Shorter-than-cap summaries are left as-is; there's no minimum."""
    words = summary.split()
    if len(words) <= max_words:
        return summary
    trimmed = " ".join(words[:max_words])
    return trimmed if trimmed.endswith((".", "?", "!")) else trimmed + "."


async def summarize_and_save_call(
    llm, context, caller_number: str, call_id: str = "", language: str = DEFAULT_LANGUAGE,
) -> None:
    """Best-effort: one extra out-of-band LLM call (pipecat's built-in
    run_inference — see GoogleLLMService) to fold this call's transcript into
    the caller's running summary, saved under the caller's number for the
    next call. Never raises — a failure here must not affect the live call,
    which has usually already ended by the time this runs.

    language: the IVR's currently configured spoken language (Bot.py's
    LANGUAGE env var, human-readable e.g. "Hindi") — the saved Name/LastTopic/
    Summary get read aloud in later greetings, so they're written in this
    language rather than whatever the deployment originally shipped with."""
    try:
        if not caller_number or not hasattr(llm, "run_inference"):
            return
        turns = [m for m in context.messages if m.get("role") in ("user", "assistant")]
        if len(turns) <= 2:  # nothing beyond the seed greeting was said
            return

        previous = await load_latest_summary(caller_number)
        previous_summary = previous.get("summary", "") if previous else ""
        previous_name = previous.get("name", "") if previous else ""
        logger.info(
            f"📝 caller_memory: summarizing call_id={call_id or 'n/a'} for {caller_number} "
            f"(building on previous call_id={previous['call_id'] if previous else 'n/a'} "
            f"at {previous['timestamp'] if previous else 'n/a'})"
        )

        transcript = _flatten_transcript(turns)
        content = (
            (f"Prior history so far:\n{previous_summary}\n\n" if previous_summary else "")
            + f"Latest call transcript:\n{transcript}"
        )
        raw = await llm.run_inference(
            LLMContext([{"role": "user", "content": content}]),
            max_tokens=SUMMARY_MAX_TOKENS,
            system_instruction=_summary_instruction(language),
        )
        if raw:
            name, last_topic, summary = _parse_name_and_summary(raw, fallback_name=previous_name)
            summary = _dedupe_repeated_runs(summary)
            summary = _cap_word_count(summary, SUMMARY_MAX_WORDS)
            await save_summary(caller_number, summary, call_id=call_id, name=name, last_topic=last_topic)
    except Exception:
        logger.opt(exception=True).warning("⚠️ caller_memory: summarization failed")


async def build_llm_greeting(
    llm, name: str, summary: str, last_topic: str = "", language: str = DEFAULT_LANGUAGE,
) -> str | None:
    """Generates a natural, name+past-topic opening line for a known
    returning caller via one extra out-of-band LLM call (same run_inference
    mechanism as summarize_and_save_call). Adds real LLM latency before the
    greeting can be spoken — see caller_memory.build_template_greeting for
    the zero-extra-latency alternative. None on any failure, so the caller
    can fall back to the static/template greeting.

    last_topic (if known — see fact_conversation_summary.last_topic) is
    passed explicitly rather than left for the model to infer from the
    summary's prose order, since the cumulative summary is topic-reorganized/
    condensed each call and is not chronological.

    language: the IVR's currently configured spoken language (Bot.py's
    LANGUAGE env var, human-readable e.g. "Hindi") — see summarize_and_save_call."""
    if not hasattr(llm, "run_inference"):
        return None
    try:
        content = f"Name: {name}\n"
        content += f"Most recent call's topic: {last_topic}\n" if last_topic else ""
        content += f"Broader past-call summary: {summary}"
        raw = await llm.run_inference(
            LLMContext([{"role": "user", "content": content}]),
            max_tokens=GREETING_MAX_TOKENS,
            system_instruction=_greeting_instruction(language),
        )
        return raw.strip() if raw else None
    except Exception:
        logger.opt(exception=True).warning("⚠️ caller_memory: LLM greeting generation failed")
        return None
