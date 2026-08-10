"""Human escalation: detects [TRANSFER_TO_AGENT] in LLM output and transfers
the live call via Tata's Call Operations API."""

import asyncio
import os

import requests
from loguru import logger

from pipecat.frames.frames import Frame, LLMFullResponseEndFrame, LLMTextFrame
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from bot_processors.calls.serializer import SmartflowFrameSerializer

ESCALATION_MARKER = "[TRANSFER_TO_AGENT]"
ESCALATION_NUMBER = os.getenv("ESCALATION_NUMBER", "")
ESCALATION_API_TOKEN = os.getenv("ESCALATION_API_TOKEN", "")
ESCALATION_FAREWELL_DELAY_SECS = float(os.getenv("ESCALATION_FAREWELL_DELAY_SECS", "5"))

# Fixed Tata Smartflo endpoint (same for every account — only the
# Authorization token above is account-specific). Docs:
# https://docs.smartflo.tatatelebusiness.com/reference/v1calloptions
TATA_CALL_OPERATIONS_URL = "https://api-smartflo.tatateleservices.com/v1/call/options"


async def _dial_escalation_number(call_id: str) -> None:
    """Transfer the live call via Tata's Call Operations API (type 4 = Transfer).

    Unlike API Dialplan (a webhook Tata calls once at initial routing), this
    is a REST call our bot makes mid-conversation the moment the LLM asks to
    escalate — call_id comes from the Smartflow 'start' event's callSid
    (see SmartflowFrameSerializer.call_id).
    """
    if not ESCALATION_API_TOKEN:
        logger.warning(
            f"⚠️ ESCALATION_API_TOKEN not set — ending call for a human at "
            f"{ESCALATION_NUMBER or '(ESCALATION_NUMBER also unset)'}, but no "
            f"transfer request was sent."
        )
        return
    if not call_id:
        logger.warning("⚠️ No call_id captured from Smartflow 'start' event — cannot transfer.")
        return
    try:
        resp = await asyncio.to_thread(
            requests.post,
            TATA_CALL_OPERATIONS_URL,
            headers={"Authorization": ESCALATION_API_TOKEN},
            json={"type": "4", "call_id": call_id, "intercom": ESCALATION_NUMBER},
            timeout=10,
        )
        logger.info(f"📞 Escalation transfer request → HTTP {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        logger.error(f"❌ Escalation transfer request failed: {e}")


class EscalationDetector(FrameProcessor):
    """Watches streamed LLM tokens for ESCALATION_MARKER and hands the call
    off to a human when the caller explicitly asks for one.

    The LLM streams LLMTextFrame token-by-token (e.g. "[", "TRANSFER", "_TO",
    "_AGENT", "]", " Sure", ...), so the marker can straddle several frames.
    This buffers the start of each LLM turn until the buffer either contains
    the full marker or can no longer be a prefix of it.

    On detection:
      1. Strips the marker so TTS only speaks the farewell that follows it.
      2. Fires _dial_escalation_number() (see above — not fully wired yet).
      3. Waits ESCALATION_FAREWELL_DELAY_SECS for the farewell audio to
         finish, then ends the call the same way the idle-timeout path does
         (task.cancel), rather than pushing a raw EndFrame.
    """

    def __init__(self, serializer: SmartflowFrameSerializer, **kwargs):
        super().__init__(**kwargs)
        # Set via set_task() once PipelineTask exists — this processor is
        # built as part of the Pipeline([...]) that PipelineTask itself wraps,
        # so the task can't be passed in at construction time.
        self._task: PipelineTask | None = None
        self._serializer = serializer
        self._buffer = ""
        self._detecting = True
        self._escalating = False
        # Set via set_finalize_callback() — lets _escalate() run the same
        # summarize/save-call/save-transcript logic on_client_disconnected
        # runs, since escalating ends the call via task.cancel() directly and
        # never fires on_client_disconnected at all.
        self._finalize_callback = None

    def set_task(self, task: PipelineTask) -> None:
        self._task = task

    def set_finalize_callback(self, callback) -> None:
        self._finalize_callback = callback

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseEndFrame):
            # True end of this LLM turn. Flush any leftover buffered prefix
            # (e.g. a lone "[" that never resolved) as real text instead of
            # dropping it, then re-arm detection for the next turn.
            if self._buffer:
                logger.debug(f"Escalation buffer flushed at response end: {self._buffer!r}")
                await self.push_frame(LLMTextFrame(self._buffer), direction)
                self._buffer = ""
            self._detecting = True
            await self.push_frame(frame, direction)
            return

        if not isinstance(frame, LLMTextFrame):
            # Other control frames (e.g. LLMFullResponseStartFrame) can land
            # mid-stream without ending the turn — pass through untouched so
            # a marker prefix split across frames by one of these still
            # resolves correctly instead of being discarded here.
            await self.push_frame(frame, direction)
            return

        if not self._detecting or not frame.text:
            await self.push_frame(frame, direction)
            return

        self._buffer += frame.text

        if ESCALATION_MARKER in self._buffer:
            self._detecting = False
            clean = self._buffer.replace(ESCALATION_MARKER, "").strip()
            self._buffer = ""
            logger.info(f"🔀 ESCALATION DETECTED — transferring call to {ESCALATION_NUMBER!r}")
            if not self._escalating:
                self._escalating = True
                asyncio.create_task(self._escalate())
            if clean:
                frame.text = clean
                await self.push_frame(frame, direction)
            return

        if len(self._buffer) < len(ESCALATION_MARKER) and ESCALATION_MARKER.startswith(self._buffer.lstrip()):
            # Still an ambiguous prefix ("[", "[TRANS", ...) — hold back
            # until it resolves one way or the other.
            return

        # Ruled out — flush the buffered text verbatim and stop inspecting
        # this LLM turn.
        self._detecting = False
        frame.text = self._buffer
        self._buffer = ""
        await self.push_frame(frame, direction)

    async def _escalate(self):
        # Wait for the farewell TTS to actually play before dialing — the
        # transfer redirects the live call leg away from our bot immediately
        # on success, so dialing first (as this used to) can cut the farewell
        # audio off mid-sentence or before it starts.
        await asyncio.sleep(ESCALATION_FAREWELL_DELAY_SECS)
        await _dial_escalation_number(self._serializer.call_id)
        if self._task is None:
            logger.error("Escalation triggered before PipelineTask was set — cannot end call")
            return
        if self._finalize_callback is not None:
            await self._finalize_callback("escalated to human agent")
        logger.info("📴 Ending call for human transfer")
        await self._task.cancel(reason="escalated to human agent")
