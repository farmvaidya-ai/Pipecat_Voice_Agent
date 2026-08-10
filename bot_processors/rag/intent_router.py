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


def classify_intent(text: str) -> str:
    """Returns "weather", "price", "location", or "other" for a caller's
    turn text. "location" (a caller dictating pincode digits) is treated
    identically to "weather"/"price" by rag.py — all three skip the RAG
    search."""
    if not text or not text.strip():
        return "other"

    if _is_digit_fragment(text):
        return "location"

    if _contains_any(text, _WEATHER_KEYWORDS):
        return "weather"

    if _contains_any(text, _PRICE_KEYWORDS) and not _contains_any(text, _PRODUCT_KEYWORDS):
        return "price"

    return "other"
