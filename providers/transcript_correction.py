"""
Post-STT dictionary correction — a safety net alongside Soniox's native
context/vocabulary biasing (see stt_factory.py).

Context biasing reduces mishearings at the ASR level, but Soniox is a closed
model — it doesn't always take the hint, especially on multi-word phrases.
This module catches what slips through: it rescans the *final* transcript
against the same domain-term list (providers/domain_vocab.py) using a
combined fuzzy (edit-distance) + phonetic match, and rewrites any close-but-
wrong span to its canonical spelling before the text reaches RAG or the LLM.

Deliberately conservative: it will NOT catch mishearings that share almost no
sound with the correct term (e.g. Bug #3's "ఎగ్ రిజల్ట్స్" vs "అగ్రి ఫెరో
సొల్యూషన్స్" — different word count, low phonetic overlap). That class of
error is what context biasing at the ASR level is for. This module is for the
more common case: small pronunciation/spelling drift on an otherwise-correct
attempt (e.g. వేరుశనగ heard as వేరుసెనగ, or ఫెరమోన్ heard as ఫిరమోన్).
"""

import re
from collections import defaultdict

from rapidfuzz import fuzz

from .domain_vocab import DOMAIN_TERMS

MATCH_THRESHOLD = 82.0
MIN_WINDOW_CHARS = 3

# Phonetic-key stripping (vowels/gemination collapsed) can make two genuinely
# unrelated short phrases coincidentally collapse to similar consonant
# skeletons (e.g. "కి ప్రైసింగ్" vs "కొమ్ము పురుగు" scored 83 phonetically
# despite only 40 raw similarity, wrongly rewriting "pricing" to "horn
# beetle"). Real drift cases (వేరుసెనగ vs వేరుశనగ, ఫిరమోన్ vs ఫెరమోన్) score
# 80+ on raw similarity too, so requiring a raw floor before trusting the
# phonetic score filters out these false positives without affecting genuine
# corrections.
RAW_SCORE_FLOOR = 55.0

# Separately: short 1-word windows can collapse to a 2-3 character phonetic
# skeleton where fuzz.ratio is unreliable (any small overlap scores high).
# E.g. the common sentence-tag "కదా" (colloquial "...right?") and the crop
# name "కంది" (pigeon pea) both strip down to "కద" — a coincidental 100%
# phonetic match despite being unrelated words, which silently injected a
# random crop name mid-sentence. Genuine drift pairs (వేరుసెనగ/వేరుశనగ,
# ఫిరమోన్/ఫెరమోన్) collapse to keys of 4+ characters, so requiring that
# length before trusting the phonetic score filters this class out too.
MIN_PHONETIC_KEY_CHARS = 4

_MAX_TERM_WORDS = max(len(t.split(" ")) for t in DOMAIN_TERMS)

# Terms grouped by word count. A window is only ever compared against terms
# of its own length — otherwise fuzz.ratio's tolerance for a strong partial
# match lets an oversized window (e.g. "స్టిక్కీ ట్రాప్స్ కోసం", 3 words)
# collapse onto a shorter term ("స్టిక్కీ ట్రాప్స్", 2 words), silently
# deleting the extra trailing word. Matching across word counts was never
# the intent (see module docstring — cross-length drift is out of scope).
_TERMS_BY_WORD_COUNT: dict[int, list[str]] = defaultdict(list)
for _term in DOMAIN_TERMS:
    _TERMS_BY_WORD_COUNT[len(_term.split(" "))].append(_term)

# Telugu vowel signs (matras) + virama + anusvara/visarga — stripped so vowel
# drift (ఫెరమోన్ vs ఫిరమోన్) collapses to the same phonetic key.
_VOWEL_MARKS = re.compile(r"[ా-ౌ్ంః]")

# Aspirated → unaspirated, and sibilant/nasal collapse — these are the
# consonant confusions most common in transliteration-driven STT drift.
# Retroflex/dental consonants are deliberately NOT merged (వరి vs వడి are
# genuinely different words) — only sounds that are near-identical in
# casual Telugu speech are collapsed here.
_CONSONANT_MAP = str.maketrans({
    "ఖ": "క", "ఘ": "గ", "ఛ": "చ", "ఝ": "జ",
    "ఠ": "ట", "ఢ": "డ", "థ": "త", "ధ": "ద", "ఫ": "ప", "భ": "బ",
    "శ": "స", "ష": "స",
    "ణ": "న", "ఙ": "న", "ఞ": "న",
})


def _phonetic_key(text: str) -> str:
    stripped = _VOWEL_MARKS.sub("", text)
    mapped = stripped.translate(_CONSONANT_MAP)
    # collapse gemination (doubled consonants, e.g. ర్ర -> ర after virama strip)
    collapsed = re.sub(r"(.)\1+", r"\1", mapped)
    return collapsed


def _similarity(window: str, term: str) -> float:
    raw_score = fuzz.ratio(window, term)
    if raw_score < RAW_SCORE_FLOOR:
        return raw_score
    phon_w, phon_t = _phonetic_key(window), _phonetic_key(term)
    if len(phon_w) < MIN_PHONETIC_KEY_CHARS or len(phon_t) < MIN_PHONETIC_KEY_CHARS:
        return raw_score
    return max(raw_score, fuzz.ratio(phon_w, phon_t))


def correct_transcript(text: str, threshold: float = MATCH_THRESHOLD) -> str:
    """
    Rewrite any span of `text` that closely (fuzzy + phonetic) matches a
    domain term to that term's canonical spelling. Returns `text` unchanged
    if nothing crosses the threshold.
    """
    words = text.split(" ")
    n = len(words)
    if n == 0:
        return text

    candidates = []  # (score, start, end, canonical_term)
    for start in range(n):
        for length in range(1, _MAX_TERM_WORDS + 1):
            end = start + length
            if end > n:
                break
            window = " ".join(words[start:end])
            if len(window) < MIN_WINDOW_CHARS:
                continue

            best_term = None
            best_score = 0.0
            for term in _TERMS_BY_WORD_COUNT.get(length, ()):
                score = _similarity(window, term)
                if score > best_score:
                    best_score = score
                    best_term = term

            if best_score >= threshold and best_term != window:
                candidates.append((best_score, start, end, best_term))

    if not candidates:
        return text

    # Greedy: take highest-scoring matches first, skip anything overlapping
    # a word index already claimed.
    candidates.sort(key=lambda c: -c[0])
    used = [False] * n
    chosen = []
    for score, start, end, term in candidates:
        if any(used[start:end]):
            continue
        for i in range(start, end):
            used[i] = True
        chosen.append((start, end, term))

    chosen.sort(key=lambda c: c[0])
    out_words = []
    i = 0
    chosen_by_start = {c[0]: c for c in chosen}
    while i < n:
        hit = chosen_by_start.get(i)
        if hit:
            _, end, term = hit
            out_words.append(term)
            i = end
        else:
            out_words.append(words[i])
            i += 1

    return " ".join(out_words)
