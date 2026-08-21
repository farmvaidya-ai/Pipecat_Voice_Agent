"""
One-off: clone a voice on Soniox from a reference audio clip.

Usage:
    python scripts/soniox_clone_voice.py --file path/to/clip.wav --name "Praneeth"

Requirements for the reference clip (per Soniox docs):
    - Single speaker, clean audio, minimal background noise
    - Up to 20 seconds (longer clips are rejected with voice_audio_too_long)
    - Max 10 MB
    - Consistent tone/pace — the model replicates pace, pitch, accent, even
      breathing/mouth clicks, so avoid highly dynamic delivery

On success, prints the voice ID. Put that ID in .env as SONIOX_TTS_VOICE
(the `voice` field accepts either a built-in voice name or a cloned voice's
UUID — same field, no other code changes needed).
"""

import argparse
import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Clone a voice on Soniox from a reference clip.")
    parser.add_argument("--file", required=True, help="Path to reference audio clip (wav/mp3/etc., <=20s, <=10MB)")
    parser.add_argument("--name", required=True, help="Unique name for this voice within your Soniox project")
    args = parser.parse_args()

    if not os.path.isfile(args.file):
        print(f"ERROR: file not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    size_mb = os.path.getsize(args.file) / (1024 * 1024)
    if size_mb > 10:
        print(f"ERROR: file is {size_mb:.1f} MB — Soniox max is 10 MB", file=sys.stderr)
        sys.exit(1)

    api_key = os.getenv("SONIOX_TTS_API_KEY") or os.getenv("SONIOX_API_KEY")
    if not api_key:
        print("ERROR: SONIOX_TTS_API_KEY / SONIOX_API_KEY not set in .env", file=sys.stderr)
        sys.exit(1)

    print(f"Uploading '{args.file}' ({size_mb:.2f} MB) as voice '{args.name}'...")

    with open(args.file, "rb") as f:
        resp = requests.post(
            "https://api.soniox.com/v1/voices",
            headers={"Authorization": f"Bearer {api_key}"},
            data={"name": args.name},
            files={"file": (os.path.basename(args.file), f)},
            timeout=60,
        )

    if resp.status_code not in (200, 201):
        print(f"ERROR: Soniox API returned {resp.status_code}: {resp.text}", file=sys.stderr)
        sys.exit(1)

    data = resp.json()
    voice_id = data["id"]

    print("\nVoice created successfully:")
    print(f"  id:      {voice_id}")
    print(f"  name:    {data.get('name')}")
    print(f"  models:  {data.get('models')}")
    print(f"\nSet in .env:\n  SONIOX_TTS_VOICE={voice_id}")


if __name__ == "__main__":
    main()
