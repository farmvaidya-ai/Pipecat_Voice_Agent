"""
One-off: synthesize a line with Bakbak's "m1" model voices so they can be
listened to side by side (and compared against the project's current
"standard" model voice, Tanvi).

Talks to the Bakbak HTTP endpoint directly (same protocol
providers/bakbak_tts.py uses — POST {text, voice_id, model, language, codec,
sample_rate} with an X-API-Key header) so it needs nothing beyond requests/
python-dotenv, already in venv.

Usage:
    python scripts/bakbak_test_voices.py
    python scripts/bakbak_test_voices.py --voices "Anika M1,Raj M1"
    python scripts/bakbak_test_voices.py --text "Namaste, yeh ek test hai."

m1 model voices, pulled live from GET https://hub.getraya.app/v1/voices
on 2026-08-18 (11 total — 9 Hindi, 1 Kannada, 1 English):
    Anika M1   (hi)  Raj M1     (hi)  Kavya M1   (kn)  Dhruvi M1  (hi)
    Anjura M1  (hi)  Riya M1    (hi)  Alice M1   (en)  Shreya M1  (hi)
    Rahul M1   (hi)  Priya M1   (hi)  Chaya M1   (hi)

Output: one WAV file per voice under scripts/voice_samples/bakbak_<name>.wav
"""

import argparse
import os
import wave

import requests
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

API_URL = "https://hub.getraya.app/v1/text-to-speech"
SAMPLE_RATE = 24000

# name -> (voice_id, language). Pulled live from GET /v1/voices (model=m1).
M1_VOICES: dict[str, tuple[str, str]] = {
    "Anika M1":  ("009ae8b5-4a03-492d-af1c-3055989ac734", "hi"),
    "Raj M1":    ("029d327b-987b-4879-8375-454cba8424ec", "hi"),
    "Kavya M1":  ("0ffd8c73-7c39-4b22-92f4-ebf33a6fae2c", "kn"),
    "Dhruvi M1": ("27ecee77-5bba-4894-9800-5be26c78837d", "hi"),
    "Anjura M1": ("3fe4afbc-3bde-4c97-ab8e-37e3fb8c7ba2", "hi"),
    "Riya M1":   ("4220e163-90d3-4fde-8faa-0e464283dc2a", "hi"),
    "Alice M1":  ("7e04362e-096f-47ad-a3b3-3a2efcbe62bc", "en"),
    "Shreya M1": ("82e802d6-d90e-4065-a1cd-a74113fb52d1", "hi"),
    "Rahul M1":  ("9c291ec7-81ce-4125-b206-80dbaacc858a", "hi"),
    "Priya M1":  ("d14c3aa6-4f90-4c11-be8f-2633a8c183b4", "hi"),
    "Chaya M1":  ("e38b1e16-9aee-4ab4-8a7e-2e7f7e51bb08", "hi"),
}

DEFAULT_VOICES = list(M1_VOICES.keys())

_DEFAULT_TEXT_BY_LANG = {
    "hi": "Namaste, yeh Soniox aur Bakbak awaazon ka ek test hai.",
    "kn": "Namaskara, ivu dhwani parikshegagi.",
    "en": "Hello, this is a quick test of the Bakbak m1 model voice.",
}

OUT_DIR = os.path.join(os.path.dirname(__file__), "voice_samples")


def synthesize_voice(api_key: str, voice_id: str, model: str, text: str, language: str) -> bytes:
    """POST one chunk to Bakbak and return raw PCM s16le audio (codec=pcm)."""
    payload = {
        "text": text,
        "voice_id": voice_id,
        "model": model,
        "language": language,
        "codec": "pcm",
        "sample_rate": SAMPLE_RATE,
    }
    resp = requests.post(
        API_URL,
        json=payload,
        headers={"X-API-Key": api_key, "Content-Type": "application/json"},
        timeout=15,
    )
    content_type = resp.headers.get("Content-Type", "")
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
    if not resp.content or len(resp.content) < 200 or "json" in content_type.lower():
        raise RuntimeError(f"non-audio response (ct={content_type!r}): {resp.text[:300]}")
    return resp.content


def save_wav(path: str, pcm_bytes: bytes):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # s16le
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_bytes)


def main():
    parser = argparse.ArgumentParser(description="Render sample audio for Bakbak m1 model voices.")
    parser.add_argument("--voices", default=",".join(DEFAULT_VOICES), help="Comma-separated voice names (must match M1_VOICES keys)")
    parser.add_argument("--text", default="", help="Text to synthesize (default: per-language sample line)")
    parser.add_argument("--model", default="m1", help="Bakbak model")
    args = parser.parse_args()

    api_key = os.getenv("BAKBAK_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: BAKBAK_API_KEY not set in .env")

    voices = [v.strip() for v in args.voices.split(",") if v.strip()]
    unknown = [v for v in voices if v not in M1_VOICES]
    if unknown:
        raise SystemExit(f"ERROR: unknown voice name(s): {unknown}. Known: {list(M1_VOICES)}")

    os.makedirs(OUT_DIR, exist_ok=True)
    logger.info(f"Testing {len(voices)} voice(s): {', '.join(voices)}  (model={args.model})")

    for name in voices:
        voice_id, language = M1_VOICES[name]
        text = args.text or _DEFAULT_TEXT_BY_LANG.get(language, _DEFAULT_TEXT_BY_LANG["en"])
        try:
            logger.info(f"  → synthesizing '{name}' (lang={language})...")
            pcm = synthesize_voice(api_key, voice_id, args.model, text, language)
            safe_name = name.replace(" ", "_")
            out_path = os.path.join(OUT_DIR, f"bakbak_{safe_name}.wav")
            save_wav(out_path, pcm)
            logger.info(f"    saved {out_path} ({len(pcm)} bytes)")
        except Exception as e:
            logger.error(f"    FAILED for voice '{name}': {e}")

    logger.info(f"Done. Listen to the files in {OUT_DIR}")


if __name__ == "__main__":
    main()
