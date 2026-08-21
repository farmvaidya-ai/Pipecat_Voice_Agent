"""Cheap upfront intent classification for a caller's turn text.

This exists purely so bot_processors/rag.py can skip firing its Qdrant
search on turns that are clearly about weather or mandi/market price,
instead of searching unconditionally on every turn regardless of what the
caller actually asked. It is NOT what decides whether get_weather/get_price
get called — that stays the LLM's own judgment (see
bot_processors/weather_lookup.py, bot_processors/price_lookup.py), which
also checks whether the caller has actually named a location/commodity/
state/district. A "other" verdict here just means "don't skip RAG" — it is
not a claim that the turn isn't a price/weather question.

A caller dictating a pincode digit-by-digit ("5, 3, 3" then "4, 0, 1") also
skips RAG, same as weather/price — confirmed live 2026-08-04
(call_919390427476, CAGE3-T9-1785836414.599878): fragments like "533.",
"40.", "0, 1." classified as "other" and each fired a real Qdrant search;
one of those (on generic filler text between pincode fragments) surfaced an
unrelated chunk about mango sticky traps, which the LLM then hallucinated
the caller had asked about. Digit-dominant fragments carry no topical
content a KB search could meaningfully match anyway.

Deliberately keyword-based rather than another LLM call: this has to run
synchronously on every interim STT fragment (see rag.py's speculative
search), so it needs to be near-instant, not add its own inference latency.

Price/rate keywords ("ధర", "price", "rate", ...) are ambiguous on their own
— a caller can just as easily ask "ఈ ట్రాప్ ధర ఎంత" (what does this trap
cost?), a legitimate product-price question that the knowledge base (not
Agmarknet) should answer. To avoid wrongly starving that turn of RAG
context, a price-keyword match is only treated as market-price intent when
none of AgriPhero's own product/category terms also appear in the same
text — those are the one place a "price" question is reliably NOT about a
mandi commodity.
"""

_WEATHER_KEYWORDS = {
    # Telugu
    "వాతావరణం", "వాతావరణం", "వర్షం", "వర్షాలు", "వాన", "వానలు", "ఎండ", "గాలి",
    "తుఫాను", "తుఫానులు", "ఉష్ణోగ్రత", "మేఘం", "మేఘాలు",
    # Telugu-script transliteration of the English word — callers code-switch
    # and say this at least as often as వాతావరణం itself. Confirmed live
    # 2026-08-05 through 2026-08-11 across 6 separate calls (e.g.
    # call_919390427476_2f3119bd.log: "సరే, అక్కడ వెదర్ ఎట్లుంది?"), every one
    # of which fell through to "other" and fired a real Qdrant search.
    "వెదర్",
    # Hindi
    "मौसम", "बारिश", "बरसात", "आंधी", "तूफान", "तापमान", "हवा",
    # Tamil
    "வானிலை", "மழை", "காற்று", "புயல்", "வெப்பநிலை",
    # Kannada
    "ಹವಾಮಾನ", "ಮಳೆ", "ಗಾಳಿ", "ಚಂಡಮಾರುತ", "ಉಷ್ಣಾಂಶ",
    # English (STT sometimes leaves these untranslated even mid-Telugu)
    "weather", "rain", "rainfall", "temperature", "humidity", "storm", "wind",
}

_PRICE_KEYWORDS = {
    # Telugu
    "ధర", "ధరలు", "రేటు", "రేట్లు", "మార్కెట్", "బజారు", "క్వింటాల్", "మండి",
    # Telugu-script transliteration of the English word, same rationale as
    # వెదర్ above — confirmed live 2026-08-11 (call_919949070894_a3fc5ca5.log:
    # "కాటన్ ప్రైస్ చెప్పండి." and call_919390427476_e8ba8f16.log: "...బ్రింజాల
    # ప్రైస్ ఎంత అనుకుని."), both fell through to "other".
    "ప్రైస్",
    # Colloquial "how much, tell me" phrasing that skips ధర/రేటు entirely
    # (e.g. "గ్రీన్ చిల్లీ ఎంత చెప్తావా?") — confirmed live 2026-08-03
    # (call_919390427476_244b1ab7.log): this fell through to "other", which
    # fed the RAG "nothing found, stay in character" marker into context
    # right as the caller finished giving commodity+state+district, and the
    # model never called get_price at all for that turn.
    "ఎంత చెప్తావా", "ఎంత చెప్పు", "ఎంత చెప్పండి", "ఎంత ఉంది", "ఎంత ఉందో",
    # Hindi
    "भाव", "कीमत", "दाम", "मंडी", "रेट",
    # Tamil
    "விலை", "சந்தை", "மார்க்கெட்",
    # Kannada
    "ಬೆಲೆ", "ಮಾರುಕಟ್ಟೆ", "ದರ",
    # English
    "price", "rate", "mandi", "market price", "cost", "selling for",
}

# AgriPhero's own product/category vocabulary (from system_prompt.txt's
# opening line and its trap-recommendation rules) — small and stable by
# nature, since it's this one business's product line, not an open-ended
# agricultural reference like Agmarknet's commodity list.
_PRODUCT_KEYWORDS = {
    "ట్రాప్", "ట్రాప్స్", "ఫెరోమోన్", "ల్యూర్", "ల్యూర్స్", "సోలార్ ట్రాప్",
    "స్టిక్కీ షీట్", "ఫ్రూట్ ఫ్లై ట్రాప్", "బకేట్ ట్రాప్", "రైనోల్యూర్",
    "trap", "traps", "pheromone", "lure", "lures", "sticky sheet", "solar trap",
}


def _contains_any(text: str, keywords: set[str]) -> bool:
    lowered = text.lower()
    return any(kw in text or kw.lower() in lowered for kw in keywords)


# Punctuation/separators a caller's dictated digits commonly get STT-split
# with ("533-401", "5, 3, 3", "0, 1.") — stripped before checking whether
# what's left is just digits.
_DIGIT_SEPARATORS = " .,-—/"


def _is_digit_fragment(text: str) -> bool:
    """True for short utterances that are mostly/only digits — a caller
    reciting a pincode a few digits at a time, not a real question."""
    stripped = text.strip()
    if not stripped or len(stripped) > 20:
        return False
    core = "".join(ch for ch in stripped if ch not in _DIGIT_SEPARATORS)
    if not core:
        return False
    digit_count = sum(ch.isdigit() for ch in core)
    return digit_count >= max(1, len(core) - 1)  # allow one stray misheard char, e.g. "I-33"


# A short caller turn with this few words is either backchannel/filler
# ("సరే.", "ఓకే."), a bare confirmation ("దాన్ని కావాలి." — "I want that"), a
# short context-dependent follow-up ("అదే, ఎప్పుడు?" — "that, when?"), a bare
# commodity name given on its own ("రాగిది." — "that's ragi"), or trailing
# off mid-thought ("ఆ, నాకు—" — "uh, I—") — none of these carry distinct
# topical content of their own for a KB search to meaningfully match against.
_LOW_CONTENT_MAX_WORDS = 4


def _is_low_content_fragment(text: str) -> bool:
    """True for a short utterance unlikely to be a real, self-contained
    question — searching it anyway risks a spurious match on whatever
    generic word is left over. Product questions ("ట్రాప్ ధర ఎంత?") are
    exempted via _PRODUCT_KEYWORDS even when short, since those do need a
    real search.

    Confirmed live 2026-08-11 (call_919949070894_a3fc5ca5.log): in one
    ~7-minute price-lookup call — entirely about mandi prices by date, never
    once actually asking about a product — 7 separate short filler/
    follow-up turns ("సరే.", "ఓకే.", "దాన్ని కావాలి.", "అదే, ఎప్పుడు?",
    "రాగిది.", "ఓకే. అలా అయితే నాకు—", "సరే, బాయ్.") each independently fired
    a real Qdrant search and injected 1-3 KB chunks (including product
    chunks — mango-fruit-fly-trap and sticky-roll pricing details — the
    caller never asked about in that turn or anywhere else in the call),
    landing in context right before the bot's next reply and surfacing as
    the bot bringing up things unprompted."""
    words = [w for w in text.strip().replace(",", " ").split() if w]
    if not words or len(words) > _LOW_CONTENT_MAX_WORDS:
        return False
    return not _contains_any(text, _PRODUCT_KEYWORDS)


def classify_intent(text: str) -> str:
    """Returns "weather", "price", "location", "fragment", or "other" for a
    caller's turn text. "location" (a caller dictating pincode digits) and
    "fragment" (a short, low-content utterance — filler, a bare
    confirmation, or trailing off mid-thought) are treated identically to
    "weather"/"price" by rag.py — all four skip the RAG search.

    Weather/price keywords are checked before the low-content fragment
    check so a short-but-real question ("ధర ఎంత?") is still labeled by
    what it actually is rather than falling into the generic "fragment"
    bucket — both skip RAG the same way either way, this is purely so the
    logs stay legible."""
    if not text or not text.strip():
        return "other"

    if _is_digit_fragment(text):
        return "location"

    if _contains_any(text, _WEATHER_KEYWORDS):
        return "weather"

    if _contains_any(text, _PRICE_KEYWORDS) and not _contains_any(text, _PRODUCT_KEYWORDS):
        return "price"

    if _is_low_content_fragment(text):
        return "fragment"

    return "other"
