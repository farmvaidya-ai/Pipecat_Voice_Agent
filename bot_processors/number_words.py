"""TTS text sanitizer: strips markdown and spells out digits/currency/percent
as spoken number words, per active call language.

system_prompt.txt tells the LLM to avoid markdown and spell out numbers in
Telugu words (rules 7-9), but the LLM doesn't reliably follow that on its
own — this enforces it in code instead of trusting the prompt.
"""

import re

from pipecat.frames.frames import Frame, LLMFullResponseEndFrame, LLMTextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transcriptions.language import Language

_TELUGU_ONES = ["", "ఒకటి", "రెండు", "మూడు", "నాలుగు", "ఐదు", "ఆరు", "ఏడు", "ఎనిమిది", "తొమ్మిది"]
_TELUGU_TEENS = {
    10: "పది", 11: "పదకొండు", 12: "పన్నెండు", 13: "పదమూడు", 14: "పద్నాలుగు",
    15: "పదిహేను", 16: "పదహారు", 17: "పదిహేడు", 18: "పద్దెనిమిది", 19: "పంతొమ్మిది",
}
_TELUGU_TENS = {
    2: "ఇరవై", 3: "ముప్పై", 4: "నలభై", 5: "యాభై",
    6: "అరవై", 7: "డెబ్బై", 8: "ఎనభై", 9: "తొంభై",
}


def _two_digit_words(n: int) -> str:
    if n == 0:
        return ""
    if n < 10:
        return _TELUGU_ONES[n]
    if n < 20:
        return _TELUGU_TEENS[n]
    tens, ones = divmod(n, 10)
    return f"{_TELUGU_TENS[tens]} {_TELUGU_ONES[ones]}".strip()


def _int_to_telugu_words(n: int) -> str:
    if n == 0:
        return "సున్నా"
    parts = []
    crore, n = divmod(n, 10**7)
    lakh, n = divmod(n, 10**5)
    thousand, n = divmod(n, 1000)
    hundred, n = divmod(n, 100)
    if crore:
        parts.append("కోటి" if crore == 1 else f"{_two_digit_words(crore)} కోట్లు")
    if lakh:
        parts.append("లక్ష" if lakh == 1 else f"{_two_digit_words(lakh)} లక్షలు")
    if thousand:
        parts.append("వెయ్యి" if thousand == 1 else f"{_two_digit_words(thousand)} వేల")
    if hundred:
        parts.append("వంద" if hundred == 1 else f"{_two_digit_words(hundred)} వందల")
    if n:
        parts.append(_two_digit_words(n))
    return " ".join(p for p in parts if p)


def _number_to_telugu_words(raw: str) -> str:
    raw = raw.replace(",", "")
    int_part, _, dec_part = raw.partition(".")
    words = _int_to_telugu_words(int(int_part or "0"))
    if dec_part:
        digit_words = " ".join(
            "సున్నా" if d == "0" else _TELUGU_ONES[int(d)] for d in dec_part
        )
        words = f"{words} పాయింట్ {digit_words}"
    return words


# ── Hindi / Tamil / Kannada / English word tables ───────────────────────────
# Same crore/lakh/thousand/hundred (Indian numbering) composition as Telugu
# above, but built through one shared generic composer since none of these
# four inflect the multiplier noun by count the way Telugu does (Telugu says
# "వందల" vs "వంద" depending on count; these just reuse the same word, e.g.
# English "two hundred" / "one hundred" — "hundred" never changes).
# Hindi numbers 1-99 are NOT tens+ones compositional (21 is "इक्कीस", not
# "बीस एक"), so Hindi gets a full irregular lookup table instead of a
# two_digit composer.

def _make_two_digit_fn(ones: list[str], teens: dict[int, str], tens: dict[int, str]):
    def fn(n: int) -> str:
        if n == 0:
            return ""
        if n < 10:
            return ones[n]
        if n < 20:
            return teens[n]
        t, o = divmod(n, 10)
        return f"{tens[t]} {ones[o]}".strip()
    return fn


def _compose_number_words(n: int, two_digit_fn, zero_word: str, hundred_word: str,
                           thousand_word: str, lakh_word: str, crore_word: str) -> str:
    if n == 0:
        return zero_word
    parts = []
    crore, n = divmod(n, 10**7)
    lakh, n = divmod(n, 10**5)
    thousand, n = divmod(n, 1000)
    hundred, n = divmod(n, 100)
    if crore:
        parts.append(f"{two_digit_fn(crore)} {crore_word}".strip() if crore > 1 else crore_word)
    if lakh:
        parts.append(f"{two_digit_fn(lakh)} {lakh_word}".strip() if lakh > 1 else lakh_word)
    if thousand:
        parts.append(f"{two_digit_fn(thousand)} {thousand_word}".strip() if thousand > 1 else thousand_word)
    if hundred:
        parts.append(f"{two_digit_fn(hundred)} {hundred_word}".strip() if hundred > 1 else hundred_word)
    if n:
        parts.append(two_digit_fn(n))
    return " ".join(p for p in parts if p)


def _make_number_to_words_fn(two_digit_fn, ones: list[str], zero_word: str, point_word: str,
                              hundred_word: str, thousand_word: str, lakh_word: str, crore_word: str):
    def fn(raw: str) -> str:
        raw = raw.replace(",", "")
        int_part, _, dec_part = raw.partition(".")
        words = _compose_number_words(
            int(int_part or "0"), two_digit_fn, zero_word, hundred_word, thousand_word, lakh_word, crore_word
        )
        if dec_part:
            digit_words = " ".join(zero_word if d == "0" else ones[int(d)] for d in dec_part)
            words = f"{words} {point_word} {digit_words}"
        return words
    return fn


# English (Indian numbering — crore/lakh, matches how these amounts are
# actually spoken in this business/farming context)
_EN_ONES = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
_EN_TEENS = {
    10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
    15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen",
}
_EN_TENS = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty", 6: "sixty", 7: "seventy", 8: "eighty", 9: "ninety"}
_en_two_digit = _make_two_digit_fn(_EN_ONES, _EN_TEENS, _EN_TENS)
_number_to_english_words = _make_number_to_words_fn(
    _en_two_digit, _EN_ONES, "zero", "point", "hundred", "thousand", "lakh", "crore"
)

# Tamil
_TA_ONES = ["", "ஒன்று", "இரண்டு", "மூன்று", "நான்கு", "ஐந்து", "ஆறு", "ஏழு", "எட்டு", "ஒன்பது"]
_TA_TEENS = {
    10: "பத்து", 11: "பதினொன்று", 12: "பன்னிரண்டு", 13: "பதிமூன்று", 14: "பதினான்கு",
    15: "பதினைந்து", 16: "பதினாறு", 17: "பதினேழு", 18: "பதினெட்டு", 19: "பத்தொன்பது",
}
_TA_TENS = {2: "இருபது", 3: "முப்பது", 4: "நாற்பது", 5: "ஐம்பது", 6: "அறுபது", 7: "எழுபது", 8: "எண்பது", 9: "தொண்ணூறு"}
_ta_two_digit = _make_two_digit_fn(_TA_ONES, _TA_TEENS, _TA_TENS)
_number_to_tamil_words = _make_number_to_words_fn(
    _ta_two_digit, _TA_ONES, "பூஜ்ஜியம்", "புள்ளி", "நூறு", "ஆயிரம்", "லட்சம்", "கோடி"
)

# Kannada
_KN_ONES = ["", "ಒಂದು", "ಎರಡು", "ಮೂರು", "ನಾಲ್ಕು", "ಐದು", "ಆರು", "ಏಳು", "ಎಂಟು", "ಒಂಬತ್ತು"]
_KN_TEENS = {
    10: "ಹತ್ತು", 11: "ಹನ್ನೊಂದು", 12: "ಹನ್ನೆರಡು", 13: "ಹದಿಮೂರು", 14: "ಹದಿನಾಲ್ಕು",
    15: "ಹದಿನೈದು", 16: "ಹದಿನಾರು", 17: "ಹದಿನೇಳು", 18: "ಹದಿನೆಂಟು", 19: "ಹತ್ತೊಂಬತ್ತು",
}
_KN_TENS = {2: "ಇಪ್ಪತ್ತು", 3: "ಮೂವತ್ತು", 4: "ನಲವತ್ತು", 5: "ಐವತ್ತು", 6: "ಅರವತ್ತು", 7: "ಎಪ್ಪತ್ತು", 8: "ಎಂಬತ್ತು", 9: "ತೊಂಬತ್ತು"}
_kn_two_digit = _make_two_digit_fn(_KN_ONES, _KN_TEENS, _KN_TENS)
_number_to_kannada_words = _make_number_to_words_fn(
    _kn_two_digit, _KN_ONES, "ಸೊನ್ನೆ", "ಪಾಯಿಂಟ್", "ನೂರು", "ಸಾವಿರ", "ಲಕ್ಷ", "ಕೋಟಿ"
)

# Malayalam (tens+ones compositional, same pattern as Tamil/Kannada above —
# native speech uses sandhi contractions at the join, e.g. "irupathiyonnu"
# for 21 rather than a clean "irupathu onnu", but the space-joined form here
# is intelligible when spoken by TTS).
_ML_ONES = ["", "ഒന്ന്", "രണ്ട്", "മൂന്ന്", "നാല്", "അഞ്ച്", "ആറ്", "ഏഴ്", "എട്ട്", "ഒൻപത്"]
_ML_TEENS = {
    10: "പത്ത്", 11: "പതിനൊന്ന്", 12: "പന്ത്രണ്ട്", 13: "പതിമൂന്ന്", 14: "പതിനാല്",
    15: "പതിനഞ്ച്", 16: "പതിനാറ്", 17: "പതിനേഴ്", 18: "പതിനെട്ട്", 19: "പത്തൊൻപത്",
}
_ML_TENS = {2: "ഇരുപത്", 3: "മുപ്പത്", 4: "നാല്പത്", 5: "അമ്പത്", 6: "അറുപത്", 7: "എഴുപത്", 8: "എൺപത്", 9: "തൊണ്ണൂറ്"}
_ml_two_digit = _make_two_digit_fn(_ML_ONES, _ML_TEENS, _ML_TENS)
_number_to_malayalam_words = _make_number_to_words_fn(
    _ml_two_digit, _ML_ONES, "പൂജ്യം", "പോയിന്റ്", "നൂറ്", "ആയിരം", "ലക്ഷം", "കോടി"
)

# Gujarati (tens+ones compositional approximation — Gujarati 21-99 actually
# use their own fused forms like Hindi, e.g. 21 is "એકવીસ" not "વીસ એક";
# this composed form is understandable but not the exact native word.
# Lower confidence than Telugu/Tamil/Kannada/Malayalam — recommend a native
# speaker review before relying on this in production.)
_GU_ONES = ["", "એક", "બે", "ત્રણ", "ચાર", "પાંચ", "છ", "સાત", "આઠ", "નવ"]
_GU_TEENS = {
    10: "દસ", 11: "અગિયાર", 12: "બાર", 13: "તેર", 14: "ચૌદ",
    15: "પંદર", 16: "સોળ", 17: "સત્તર", 18: "અઢાર", 19: "ઓગણીસ",
}
_GU_TENS = {2: "વીસ", 3: "ત્રીસ", 4: "ચાલીસ", 5: "પચાસ", 6: "સાઠ", 7: "સિત્તેર", 8: "એંસી", 9: "નેવું"}
_gu_two_digit = _make_two_digit_fn(_GU_ONES, _GU_TEENS, _GU_TENS)
_number_to_gujarati_words = _make_number_to_words_fn(
    _gu_two_digit, _GU_ONES, "શૂન્ય", "પોઈન્ટ", "સો", "હજાર", "લાખ", "કરોડ"
)

# Punjabi (Gurmukhi, tens+ones compositional approximation — same caveat as
# Gujarati above: native 21-99 are fused forms like "ਇੱਕੀ" for 21, not
# "ਵੀਹ ਇੱਕ". Lower confidence — recommend native speaker review.)
_PA_ONES = ["", "ਇੱਕ", "ਦੋ", "ਤਿੰਨ", "ਚਾਰ", "ਪੰਜ", "ਛੇ", "ਸੱਤ", "ਅੱਠ", "ਨੌਂ"]
_PA_TEENS = {
    10: "ਦਸ", 11: "ਗਿਆਰਾਂ", 12: "ਬਾਰਾਂ", 13: "ਤੇਰਾਂ", 14: "ਚੌਦਾਂ",
    15: "ਪੰਦਰਾਂ", 16: "ਸੋਲਾਂ", 17: "ਸਤਾਰਾਂ", 18: "ਅਠਾਰਾਂ", 19: "ਉੱਨੀ",
}
_PA_TENS = {2: "ਵੀਹ", 3: "ਤੀਹ", 4: "ਚਾਲੀ", 5: "ਪੰਜਾਹ", 6: "ਸੱਠ", 7: "ਸੱਤਰ", 8: "ਅੱਸੀ", 9: "ਨੱਬੇ"}
_pa_two_digit = _make_two_digit_fn(_PA_ONES, _PA_TEENS, _PA_TENS)
_number_to_punjabi_words = _make_number_to_words_fn(
    _pa_two_digit, _PA_ONES, "ਸਿਫ਼ਰ", "ਪੁਆਇੰਟ", "ਸੌ", "ਹਜ਼ਾਰ", "ਲੱਖ", "ਕਰੋੜ"
)

# Odia (tens+ones compositional approximation. Odia numerals are strongly
# irregular natively, similar to Bengali/Assamese — this composed form is
# the LOWEST-confidence table in this file. Strongly recommend a native
# speaker review before relying on this in production.)
_OR_ONES = ["", "ଏକ", "ଦୁଇ", "ତିନି", "ଚାରି", "ପାଞ୍ଚ", "ଛଅ", "ସାତ", "ଆଠ", "ନଅ"]
_OR_TEENS = {
    10: "ଦଶ", 11: "ଏଗାର", 12: "ବାର", 13: "ତେର", 14: "ଚଉଦ",
    15: "ପନ୍ଦର", 16: "ଷୋହଳ", 17: "ସତର", 18: "ଅଠର", 19: "ଉଣେଇଶ",
}
_OR_TENS = {2: "କୋଡିଏ", 3: "ତିରିଶ", 4: "ଚାଳିଶ", 5: "ପଚାଶ", 6: "ଷାଠିଏ", 7: "ସତୁରି", 8: "ଅଶୀ", 9: "ନବେ"}
_or_two_digit = _make_two_digit_fn(_OR_ONES, _OR_TEENS, _OR_TENS)
_number_to_odia_words = _make_number_to_words_fn(
    _or_two_digit, _OR_ONES, "ଶୂନ୍ୟ", "ପଏଣ୍ଟ", "ଶହେ", "ହଜାର", "ଲକ୍ଷ", "କୋଟି"
)

# Assamese (tens+ones compositional approximation — same lowest-confidence
# caveat as Odia above: natively irregular, this is an approximation.
# Recommend native speaker review before relying on this in production.)
_AS_ONES = ["", "এক", "দুই", "তিনি", "চাৰি", "পাঁচ", "ছয়", "সাত", "আঠ", "ন"]
_AS_TEENS = {
    10: "দহ", 11: "এঘাৰ", 12: "বাৰ", 13: "তেৰ", 14: "সোঁৱাৰ",
    15: "পোন্ধৰ", 16: "ষোল্ল", 17: "সোতৰ", 18: "আঠাৰ", 19: "ঊনৈশ",
}
_AS_TENS = {2: "বিশ", 3: "ত্ৰিছ", 4: "চল্লিশ", 5: "পঞ্চাশ", 6: "ষাঠি", 7: "সত্তৰ", 8: "আশী", 9: "নব্বৈ"}
_as_two_digit = _make_two_digit_fn(_AS_ONES, _AS_TEENS, _AS_TENS)
_number_to_assamese_words = _make_number_to_words_fn(
    _as_two_digit, _AS_ONES, "শূন্য", "পইণ্ট", "শ", "হাজাৰ", "লাখ", "কোটি"
)

# Hindi — 1-99 are irregular (not tens+ones compositional), so this is a
# full lookup table rather than the composer used for the other languages.
_HI_ONES = ["", "एक"]  # index 0/1 only used for decimal digits below
_HI_UNITS = [
    "शून्य", "एक", "दो", "तीन", "चार", "पांच", "छह", "सात", "आठ", "नौ", "दस",
    "ग्यारह", "बारह", "तेरह", "चौदह", "पंद्रह", "सोलह", "सत्रह", "अठारह", "उन्नीस", "बीस",
    "इक्कीस", "बाईस", "तेईस", "चौबीस", "पच्चीस", "छब्बीस", "सत्ताईस", "अट्ठाईस", "उनतीस", "तीस",
    "इकतीस", "बत्तीस", "तैंतीस", "चौंतीस", "पैंतीस", "छत्तीस", "सैंतीस", "अड़तीस", "उनतालीस", "चालीस",
    "इकतालीस", "बयालीस", "तैंतालीस", "चौंतालीस", "पैंतालीस", "छियालीस", "सैंतालीस", "अड़तालीस", "उनचास", "पचास",
    "इक्यावन", "बावन", "तिरपन", "चौवन", "पचपन", "छप्पन", "सत्तावन", "अट्ठावन", "उनसठ", "साठ",
    "इकसठ", "बासठ", "तिरसठ", "चौंसठ", "पैंसठ", "छियासठ", "सड़सठ", "अड़सठ", "उनहत्तर", "सत्तर",
    "इकहत्तर", "बहत्तर", "तिहत्तर", "चौहत्तर", "पचहत्तर", "छिहत्तर", "सतहत्तर", "अठहत्तर", "उनासी", "अस्सी",
    "इक्यासी", "बयासी", "तिरासी", "चौरासी", "पचासी", "छियासी", "सत्तासी", "अठासी", "नवासी", "नब्बे",
    "इक्यानवे", "बानवे", "तिरानवे", "चौरानवे", "पंचानवे", "छियानवे", "सत्तानवे", "अट्ठानवे", "निन्यानवे",
]
_HI_DIGIT_ONES = ["शून्य", "एक", "दो", "तीन", "चार", "पांच", "छह", "सात", "आठ", "नौ"]  # for decimal digits


def _hi_two_digit(n: int) -> str:
    return _HI_UNITS[n] if n else ""


_number_to_hindi_words = _make_number_to_words_fn(
    _hi_two_digit, _HI_DIGIT_ONES, "शून्य", "पॉइंट", "सौ", "हज़ार", "लाख", "करोड़"
)

# Marathi — 1-99 irregular (same reason as Hindi), full lookup table.
_MR_UNITS = [
    "शून्य", "एक", "दोन", "तीन", "चार", "पाच", "सहा", "सात", "आठ", "नऊ", "दहा",
    "अकरा", "बारा", "तेरा", "चौदा", "पंधरा", "सोळा", "सतरा", "अठरा", "एकोणीस", "वीस",
    "एकवीस", "बावीस", "तेवीस", "चोवीस", "पंचवीस", "सव्वीस", "सत्तावीस", "अठ्ठावीस", "एकोणतीस", "तीस",
    "एकतीस", "बत्तीस", "तेहतीस", "चौतीस", "पस्तीस", "छत्तीस", "सदतीस", "अडतीस", "एकोणचाळीस", "चाळीस",
    "एक्केचाळीस", "बेचाळीस", "त्रेचाळीस", "चव्वेचाळीस", "पंचेचाळीस", "सेहेचाळीस", "सत्तेचाळीस", "अठ्ठेचाळीस", "एकोणपन्नास", "पन्नास",
    "एक्कावन्न", "बावन्न", "त्रेपन्न", "चोपन्न", "पंचावन्न", "छप्पन्न", "सत्तावन्न", "अठ्ठावन्न", "एकोणसाठ", "साठ",
    "एकसष्ठ", "बासष्ठ", "त्रेसष्ठ", "चौसष्ठ", "पासष्ठ", "सहासष्ठ", "सदुसष्ठ", "अडुसष्ठ", "एकोणसत्तर", "सत्तर",
    "एक्काहत्तर", "बहात्तर", "त्र्याहत्तर", "चौऱ्याहत्तर", "पंच्याहत्तर", "शहात्तर", "सत्याहत्तर", "अठ्ठ्याहत्तर", "एकोणऐंशी", "ऐंशी",
    "एक्क्याऐंशी", "ब्याऐंशी", "त्र्याऐंशी", "चौऱ्याऐंशी", "पंच्याऐंशी", "शहाऐंशी", "सत्याऐंशी", "अठ्ठ्याऐंशी", "एकोणनव्वद", "नव्वद",
    "एक्क्याण्णव", "ब्याण्णव", "त्र्याण्णव", "चौऱ्याण्णव", "पंच्याण्णव", "शहाण्णव", "सत्त्याण्णव", "अठ्ठ्याण्णव", "नव्याण्णव",
]
_MR_DIGIT_ONES = ["शून्य", "एक", "दोन", "तीन", "चार", "पाच", "सहा", "सात", "आठ", "नऊ"]


def _mr_two_digit(n: int) -> str:
    return _MR_UNITS[n] if n else ""


_number_to_marathi_words = _make_number_to_words_fn(
    _mr_two_digit, _MR_DIGIT_ONES, "शून्य", "पॉइंट", "शंभर", "हजार", "लाख", "कोटी"
)

# Bengali — 1-99 irregular (same reason as Hindi), full lookup table.
_BN_UNITS = [
    "শূন্য", "এক", "দুই", "তিন", "চার", "পাঁচ", "ছয়", "সাত", "আট", "নয়", "দশ",
    "এগারো", "বারো", "তেরো", "চৌদ্দ", "পনেরো", "ষোলো", "সতেরো", "আঠারো", "উনিশ", "কুড়ি",
    "একুশ", "বাইশ", "তেইশ", "চব্বিশ", "পঁচিশ", "ছাব্বিশ", "সাতাশ", "আটাশ", "ঊনত্রিশ", "ত্রিশ",
    "একত্রিশ", "বত্রিশ", "তেত্রিশ", "চৌত্রিশ", "পঁয়ত্রিশ", "ছত্রিশ", "সাঁইত্রিশ", "আটত্রিশ", "ঊনচল্লিশ", "চল্লিশ",
    "একচল্লিশ", "বিয়াল্লিশ", "তেতাল্লিশ", "চুয়াল্লিশ", "পঁয়তাল্লিশ", "ছেচল্লিশ", "সাতচল্লিশ", "আটচল্লিশ", "ঊনপঞ্চাশ", "পঞ্চাশ",
    "একান্ন", "বাহান্ন", "তেপ্পান্ন", "চুয়ান্ন", "পঞ্চান্ন", "ছাপ্পান্ন", "সাতান্ন", "আটান্ন", "ঊনষাট", "ষাট",
    "একষট্টি", "বাষট্টি", "তেষট্টি", "চৌষট্টি", "পঁয়ষট্টি", "ছেষট্টি", "সাতষট্টি", "আটষট্টি", "ঊনসত্তর", "সত্তর",
    "একাত্তর", "বাহাত্তর", "তিয়াত্তর", "চুয়াত্তর", "পঁচাত্তর", "ছিয়াত্তর", "সাতাত্তর", "আটাত্তর", "ঊনআশি", "আশি",
    "একাশি", "বিরাশি", "তিরাশি", "চুরাশি", "পঁচাশি", "ছিয়াশি", "সাতাশি", "আটাশি", "ঊননব্বই", "নব্বই",
    "একানব্বই", "বিরানব্বই", "তিরানব্বই", "চুরানব্বই", "পঁচানব্বই", "ছিয়ানব্বই", "সাতানব্বই", "আটানব্বই", "নিরানব্বই",
]
_BN_DIGIT_ONES = ["শূন্য", "এক", "দুই", "তিন", "চার", "পাঁচ", "ছয়", "সাত", "আট", "নয়"]


def _bn_two_digit(n: int) -> str:
    return _BN_UNITS[n] if n else ""


_number_to_bengali_words = _make_number_to_words_fn(
    _bn_two_digit, _BN_DIGIT_ONES, "শূন্য", "পয়েন্ট", "শত", "হাজার", "লক্ষ", "কোটি"
)

# Dispatch: current call language → (number-words fn, currency word, percent word, range connector).
# Languages not listed here have no word table yet — the sanitizer leaves
# their digits/currency/percent as raw numerals rather than guessing.
_NUMBER_LANG_CONFIG: dict = {
    Language.TE: (_number_to_telugu_words, "రూపాయలు", "శాతం", "నుంచి"),
    Language.EN: (_number_to_english_words, "rupees", "percent", "to"),
    Language.HI: (_number_to_hindi_words, "रुपये", "प्रतिशत", "से"),
    Language.TA: (_number_to_tamil_words, "ரூபாய்", "சதவீதம்", "முதல்"),
    Language.KN: (_number_to_kannada_words, "ರೂಪಾಯಿಗಳು", "ಶೇಕಡಾ", "ರಿಂದ"),
    Language.ML: (_number_to_malayalam_words, "രൂപ", "ശതമാനം", "മുതൽ"),
    Language.MR: (_number_to_marathi_words, "रुपये", "टक्के", "ते"),
    Language.BN: (_number_to_bengali_words, "টাকা", "শতাংশ", "থেকে"),
    Language.GU: (_number_to_gujarati_words, "રૂપિયા", "ટકા", "થી"),
    Language.PA: (_number_to_punjabi_words, "ਰੁਪਏ", "ਪ੍ਰਤੀਸ਼ਤ", "ਤੋਂ"),
    Language.OR: (_number_to_odia_words, "ଟଙ୍କା", "ପ୍ରତିଶତ", "ରୁ"),
    Language.AS: (_number_to_assamese_words, "টকা", "শতাংশ", "ৰ পৰা"),
}


_CURRENCY_RE = re.compile(r'₹\s?(\d[\d,]*\.?\d*)')
_PERCENT_RE = re.compile(r'(\d[\d,]*\.?\d*)\s?%')
_RANGE_RE = re.compile(r'\b(\d[\d,]*)\s*-\s*(\d[\d,]*)\b')
_NUMBER_RE = re.compile(r'\d[\d,]*\.?\d*')

_MARKDOWN_RE = re.compile(r'\*\*|\*|__|`{1,3}|#{1,6}\s*')
_BULLET_RE = re.compile(r'(?m)^[ \t]*[-•+]\s+')

# Trailing run of characters that might still be extending into the next
# streamed frame (a number, a currency/percent marker, or a markdown token) —
# held back rather than sanitized immediately, so a number/marker split
# across two LLMTextFrames isn't corrupted by being processed too early.
_HOLDBACK_RE = re.compile(r'[\d,.%₹\-*`#]+$')


def _sanitize_tts_text(text: str, language) -> str:
    text = _BULLET_RE.sub("", text)
    text = _MARKDOWN_RE.sub("", text)
    cfg = _NUMBER_LANG_CONFIG.get(language)
    if cfg is None:
        # No word table for this language yet — leave digits as-is rather
        # than guess. Markdown stripping above still applies to every language.
        return text
    number_words, currency_word, percent_word, range_word = cfg
    text = _CURRENCY_RE.sub(lambda m: f"{number_words(m.group(1))} {currency_word}", text)
    text = _PERCENT_RE.sub(lambda m: f"{number_words(m.group(1))} {percent_word}", text)
    text = _RANGE_RE.sub(
        lambda m: f"{number_words(m.group(1))} {range_word} {number_words(m.group(2))}",
        text,
    )
    text = _NUMBER_RE.sub(lambda m: number_words(m.group(0)), text)
    return text


class TTSTextSanitizer(FrameProcessor):
    """Sits between the LLM and TTS. Always strips markdown. Additionally
    converts digits/currency/percent/ranges into spoken number words, using
    the word table for whichever language is currently active (Telugu,
    English, Hindi, Tamil, Kannada — see _NUMBER_LANG_CONFIG). This agent is
    multilingual (LANGUAGE=auto hot-switches TTS language per detected turn
    via MultilingualTTSSwitcher), so the conversion must follow the live
    call language rather than assuming a single fixed one.

    Only the trailing run matched by _HOLDBACK_RE is buffered — everything
    else is sanitized and forwarded immediately, so this doesn't undo the
    word-batch TTS aggregator's latency gains by waiting for a full sentence.
    """

    def __init__(self, lang_switcher=None, default_language=None, **kwargs):
        super().__init__(**kwargs)
        self._buffer = ""
        self._lang_switcher = lang_switcher
        self._default_language = default_language

    def _current_language(self):
        if self._lang_switcher is not None and self._lang_switcher._current_lang is not None:
            return self._lang_switcher._current_lang
        return self._default_language

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseEndFrame):
            if self._buffer:
                cleaned = _sanitize_tts_text(self._buffer, self._current_language())
                self._buffer = ""
                if cleaned:
                    await self.push_frame(LLMTextFrame(cleaned), direction)
            await self.push_frame(frame, direction)
            return

        if not isinstance(frame, LLMTextFrame) or not frame.text:
            await self.push_frame(frame, direction)
            return

        combined = self._buffer + frame.text
        match = _HOLDBACK_RE.search(combined)
        held_from = match.start() if match else len(combined)
        ready, self._buffer = combined[:held_from], combined[held_from:]

        if ready:
            cleaned = _sanitize_tts_text(ready, self._current_language())
            if cleaned:
                await self.push_frame(LLMTextFrame(cleaned), direction)
