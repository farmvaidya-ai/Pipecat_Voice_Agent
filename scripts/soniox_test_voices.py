"""
One-off: synthesize the same line with several Soniox tts-rt-v2 stock voices
so they can be listened to side by side (and compared against the project's
cloned "Praneeth" voice).

Talks to the Soniox TTS websocket directly (same protocol pipecat's
SonioxTTSService uses — see venv/Lib/site-packages/pipecat/services/soniox/tts.py)
so it needs nothing beyond requests/websockets/python-dotenv, already in venv.

Usage:
    python scripts/soniox_test_voices.py
    python scripts/soniox_test_voices.py --text "Namaste, mandi bhaav check karne ke liye dhanyavaad."
    python scripts/soniox_test_voices.py --voices Adrian,Arjun,Priya,Grace --language en

Default 4 voices — all Indian-accented, from docs/soniox_voice_roster.pdf's
70-voice tts-rt-v2 list, tested in this order:
    Arjun   - Indian male accent
    Karan   - Indian male accent
    Rohan   - Indian male accent
    Priya   - Indian female accent

Output: one WAV file per voice under scripts/voice_samples/<voice>.wav
"""

import argparse
import asyncio
import base64
import json
import os
import wave

import websockets
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

WS_URL = "wss://tts-rt.soniox.com/tts-websocket"
SAMPLE_RATE = 24000
DEFAULT_VOICES = ["Arjun", "Karan", "Rohan", "Priya"]
DEFAULT_TEXT = (
    "Hello, this is a quick test of the Soniox text to speech voice. "
    "I hope this sounds clear and natural."
)
OUT_DIR = os.path.join(os.path.dirname(__file__), "voice_samples")


async def synthesize_voice(api_key: str, voice: str, model: str, text: str, language: str) -> bytes:
    """Open one Soniox TTS stream for `voice` and return the raw PCM s16le audio."""
    audio = bytearray()
    async with websockets.connect(WS_URL) as ws:
        config = {
            "api_key": api_key,
            "stream_id": "1",
            "model": model,
            "voice": voice,
            "audio_format": "pcm_s16le",
            "sample_rate": SAMPLE_RATE,
            "language": language,
            "return_timestamps": False,
        }
        await ws.send(json.dumps(config))
        await ws.send(json.dumps({"text": text, "text_end": False, "stream_id": "1"}))
        await ws.send(json.dumps({"text": "", "text_end": True, "stream_id": "1"}))

        while True:
            raw = await ws.recv()
            msg = json.loads(raw)
            if msg.get("error_code"):
                raise RuntimeError(f"Soniox error for voice={voice}: {msg}")
            audio_b64 = msg.get("audio")
            if audio_b64:
                audio.extend(base64.b64decode(audio_b64))
            if msg.get("finished") or msg.get("terminated"):
                break
    return bytes(audio)


def save_wav(path: str, pcm_bytes: bytes):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # s16le
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_bytes)


async def main_async(voices: list[str], text: str, language: str, model: str):
    api_key = os.getenv("SONIOX_TTS_API_KEY") or os.getenv("SONIOX_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: SONIOX_TTS_API_KEY / SONIOX_API_KEY not set in .env")

    os.makedirs(OUT_DIR, exist_ok=True)
    logger.info(f"Testing {len(voices)} voice(s): {', '.join(voices)}  (model={model})")

    for voice in voices:
        try:
            logger.info(f"  → synthesizing '{voice}'...")
            pcm = await synthesize_voice(api_key, voice, model, text, language)
            out_path = os.path.join(OUT_DIR, f"{voice}.wav")
            save_wav(out_path, pcm)
            logger.info(f"    saved {out_path} ({len(pcm)} bytes)")
        except Exception as e:
            logger.error(f"    FAILED for voice '{voice}': {e}")

    logger.info(f"Done. Listen to the files in {OUT_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Render sample audio for several Soniox tts-rt-v2 voices.")
    parser.add_argument("--voices", default=",".join(DEFAULT_VOICES), help="Comma-separated voice names")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Text to synthesize")
    parser.add_argument("--language", default="en", help="ISO language code (en, hi, te, ta, ...)")
    parser.add_argument("--model", default="tts-rt-v2", help="Soniox TTS model")
    args = parser.parse_args()

    voices = [v.strip() for v in args.voices.split(",") if v.strip()]
    asyncio.run(main_async(voices, args.text, args.language, args.model))


if __name__ == "__main__":
    main()
