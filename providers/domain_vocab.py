"""
Shared Telugu domain vocabulary for the Farm Vaidya bot.

Single source of truth used by two independent fixes for STT accuracy:
  - stt_factory.py's _create_soniox()   → ASR-level fix (Soniox native context/
                                           vocabulary biasing, applied *during*
                                           transcription)
  - transcript_correction.py            → post-STT fix (fuzzy + phonetic
                                           dictionary correction, applied
                                           *after* transcription, as a safety
                                           net for whatever context biasing
                                           doesn't fully catch)
"""

CONTEXT_TEXT = (
    "వ్యవసాయ పంటలలో పురుగుల నివారణకు ఫెరమోన్ ట్రాప్స్ మరియు ల్యూర్స్ తయారు చేసే "
    "అగ్రి ఫెరో సొల్యూషన్స్ (Agri Phero Solutionz) కంపెనీకి సంబంధించిన రైతుల కాల్."
)

DOMAIN_TERMS = [
    # Brand name — all spelling variants found in KB/system_prompt + English
    "అగ్రి ఫెరో సొల్యూషన్స్", "అగ్రి ఫెర్రో సొల్యూషన్స్", "అగ్రిఫెరో సొల్యూషన్స్",
    "Agri Phero Solutionz", "Agri Phero Solutions",

    # Products / traps / lures
    "ఫెరమోన్ ట్రాప్స్", "ఫెరమోన్ లూర్స్", "ఫిరమోన్ టెక్నాలజీ", "లింగాకర్షక బుట్ట",
    "స్టిక్కీ ట్రాప్స్", "స్టిక్కీ రోల్స్", "బకెట్ ట్రాప్", "రైనోల్యూర్",
    "ఆర్‌పిడబ్ల్యూ ల్యూర్", "ఫన్నెల్ ట్రాప్", "మాక్స్ ఫిల్ ఫ్రూట్ ఫ్లై ట్రాప్",
    "ఫ్రూట్ ఫ్లై ల్యూర్", "సోలార్ లైట్ ట్రాప్",

    # Crops
    "వక్క తోట", "కొబ్బరి", "ఆయిల్ పామ్", "ఖర్జూరం", "మామిడి", "వరి", "మొక్కజొన్న",
    "పత్తి", "మిరప", "వేరుశనగ", "టమాటా", "పెసర", "మినుము", "జామ", "బొప్పాయి",
    "సీతాఫలం", "వంగ", "బెండ", "క్యాబేజీ", "ద్రాక్ష", "సోయా చిక్కుడు", "రాగి",
    "శనగ", "కంది", "ప్రొద్దు తిరుగుడు", "ఆముదం", "తమలపాకు", "క్యాప్సికం",
    "చిలగడదుంప", "చామగడ్డ",

    # Pests
    "పండు ఈగ", "కొమ్ము పురుగు", "ఎర్రముక్కు పురుగు", "కత్తెర పురుగు",
    "కాండం తొలుచు పురుగు", "పొగాకు లద్దె పురుగు", "శనగ పచ్చపురుగు",
    "గులాబీ రంగు కాయతొలుచు పురుగు", "తేనె మంచు పురుగు", "త్రిప్స్",
    "తెల్లదోమ", "పచ్చదోమ", "క్యాబేజీ రెక్కల పురుగు",
]
