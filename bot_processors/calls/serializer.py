"""Tata Smartflow WebSocket frame serializer (G.711 mu-law 8 kHz <-> PCM 16 kHz)."""

import asyncio
import audioop  # deprecated in Python 3.11+; migrate to audioop-lts or numpy when dropping <3.11
import base64
import json as _json

from loguru import logger

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    AudioRawFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    StartFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer


class SmartflowFrameSerializer(FrameSerializer):
    """Tata Smartflow WebSocket serializer (G.711 μ-law 8 kHz ↔ PCM 16 kHz)."""

    SMARTFLOW_SAMPLE_RATE = 8000

    def __init__(self):
        super().__init__(FrameSerializer.InputParams())
        self._sample_rate = 0
        self._stream_sid = ""
        self.call_id = ""
        self.caller_number = ""
        self._input_resampler = create_stream_resampler()
        self._output_resampler = create_stream_resampler()
        self._start_event = asyncio.Event()

    def reset_for_new_call(self):
        self._stream_sid = ""
        self.call_id = ""
        self.caller_number = ""
        self._start_event.clear()

    async def wait_for_start(self):
        """Wait until Smartflow's 'start' event has set the streamSid."""
        await self._start_event.wait()

    async def setup(self, frame: StartFrame):
        self._sample_rate = frame.audio_in_sample_rate

    async def serialize(self, frame: Frame) -> str | None:
        if isinstance(frame, AudioRawFrame):
            # If the greeting TTS fires before Smartflow's 'start' event sets
            # streamSid, wait here instead of blocking on_client_connected.
            # This lets TTS generation and the start-event wait run in parallel.
            if not self._stream_sid:
                await self._start_event.wait()
            resampled = await self._output_resampler.resample(
                frame.audio, frame.sample_rate, self.SMARTFLOW_SAMPLE_RATE
            )
            if not resampled:
                return None
            mulaw = audioop.lin2ulaw(bytes(resampled), 2)
            payload = base64.b64encode(mulaw).decode("ascii")
            return _json.dumps({
                "event": "media",
                "streamSid": self._stream_sid,
                "media": {"payload": payload},
            })
        if isinstance(frame, InterruptionFrame):
            return _json.dumps({"event": "clear", "streamSid": self._stream_sid})
        return None

    async def deserialize(self, data: bytes | str) -> Frame | None:
        if not isinstance(data, str):
            return None
        try:
            msg = _json.loads(data)
        except ValueError:
            logger.warning(f"[Smartflow] non-JSON: {data[:100]}")
            return None

        event = msg.get("event", "")

        if event == "start":
            self._stream_sid = msg.get("start", {}).get("streamSid", "")
            self.call_id = msg.get("start", {}).get("callSid", "")
            # Tata's 'from' is E.164-ish (e.g. "+919390427476") — keep digits
            # only, matching the bare-digit number format used everywhere else
            # in this codebase (.env's OUTBOUND_CALLER_ID, ESCALATION_NUMBER, etc).
            raw_from = msg.get("start", {}).get("from", "")
            self.caller_number = "".join(ch for ch in raw_from if ch.isdigit())
            fmt = msg.get("start", {}).get("mediaFormat", {})
            logger.info(
                f"[Smartflow] call started — streamSid={self._stream_sid} "
                f"caller={self.caller_number} "
                f"encoding={fmt.get('encoding')} sampleRate={fmt.get('sampleRate')}"
            )
            # Diagnostic: dump the full raw 'start' payload once per call, in case
            # Smartflow exposes an echo-cancellation / AEC field on the stream that
            # we're not currently reading (investigating caller-side line echo
            # reported on real handset calls — see conversation 2026-07-09).
            logger.info(f"[Smartflow] full start payload: {msg}")
            self._start_event.set()
        elif event == "media":
            payload_b64 = msg.get("media", {}).get("payload", "")
            if not payload_b64:
                return None
            mulaw = base64.b64decode(payload_b64)
            pcm = audioop.ulaw2lin(mulaw, 2)
            resampled = await self._input_resampler.resample(
                pcm, self.SMARTFLOW_SAMPLE_RATE, self._sample_rate
            )
            if not resampled:
                return None
            return InputAudioRawFrame(
                audio=bytes(resampled),
                num_channels=1,
                sample_rate=self._sample_rate,
            )
        elif event == "stop":
            logger.info(f"[Smartflow] call stopped — reason={msg.get('stop', {}).get('reason', '')}")
        elif event == "connected":
            logger.info("[Smartflow] WebSocket connection acknowledged")
        else:
            logger.debug(f"[Smartflow] unhandled event: {event}")

        return None
