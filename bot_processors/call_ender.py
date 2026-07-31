"""Caller-requested call termination: detects [END_CALL] in LLM output and
actually ends the call once the farewell has been spoken.

Before this, a caller saying "end the call" / "cut the call" only got a
verbal-sounding farewell from the LLM (e.g. "సరే, కాల్ డిస్‌కనెక్ట్
చేస్తున్నాను") with no backing action — the call just kept running,
degenerating into nonsense turns once the caller assumed it had ended
(see logs/call_919390427476_255c1f32.log, 2026-07-23). Mirrors
EscalationDetector's marker-buffering approach (see escalation.py) but with
no transfer step — just strip the marker, let the farewell TTS play, then
tear the pipeline down the same way the idle-timeout/escalation paths do.
"""

import asyncio
import os

from loguru import logger

from pipecat.frames.frames import Frame, LLMFullResponseEndFrame, LLMTextFrame
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

END_CALL_MARKER = "[END_CALL]"
END_CALL_FAREWELL_DELAY_SECS = float(os.getenv("END_CALL_FAREWELL_DELAY_SECS", "3"))


class CallEndDetector(FrameProcessor):
    """Watches streamed LLM tokens for END_CALL_MARKER and ends the call once
    the caller has explicitly asked to end/hang up/disconnect it.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set via set_task() once PipelineTask exists — same reason as
        # EscalationDetector: this processor is built as part of the
        # Pipeline([...]) that PipelineTask itself wraps.
        self._task: PipelineTask | None = None
        self._buffer = ""
        self._detecting = True
        self._ending = False
        # Set via set_finalize_callback() — runs the same summarize/save-call/
        # save-transcript logic on_client_disconnected runs, since ending the
        # call here goes straight to task.cancel() and never fires
        # on_client_disconnected at all.
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
                logger.debug(f"End-call buffer flushed at response end: {self._buffer!r}")
                await self.push_frame(LLMTextFrame(self._buffer), direction)
                self._buffer = ""
            self._detecting = True
            await self.push_frame(frame, direction)
            return

        if not isinstance(frame, LLMTextFrame):
            # Other control frames (e.g. LLMFullResponseStartFrame) can land
            # mid-stream without ending the turn — pass through untouched.
            await self.push_frame(frame, direction)
            return

        if not self._detecting or not frame.text:
            await self.push_frame(frame, direction)
            return

        self._buffer += frame.text

        if END_CALL_MARKER in self._buffer:
            self._detecting = False
            clean = self._buffer.replace(END_CALL_MARKER, "").strip()
            self._buffer = ""
            logger.info("📴 END CALL DETECTED — caller asked to end the call")
            if not self._ending:
                self._ending = True
                asyncio.create_task(self._end())
            if clean:
                frame.text = clean
                await self.push_frame(frame, direction)
            return

        if len(self._buffer) < len(END_CALL_MARKER) and END_CALL_MARKER.startswith(self._buffer.lstrip()):
            # Still an ambiguous prefix ("[", "[END", ...) — hold back until
            # it resolves one way or the other.
            return

        # Ruled out — flush the buffered text verbatim and stop inspecting
        # this LLM turn.
        self._detecting = False
        frame.text = self._buffer
        self._buffer = ""
        await self.push_frame(frame, direction)

    async def _end(self):
        # Wait for the farewell TTS to actually play before tearing the
        # pipeline down — same reasoning as EscalationDetector's _escalate().
        await asyncio.sleep(END_CALL_FAREWELL_DELAY_SECS)
        if self._task is None:
            logger.error("End-call triggered before PipelineTask was set — cannot end call")
            return
        if self._finalize_callback is not None:
            await self._finalize_callback("caller asked to end the call")
        logger.info("📴 Ending call at caller's request")
        await self._task.cancel(reason="caller asked to end the call")
