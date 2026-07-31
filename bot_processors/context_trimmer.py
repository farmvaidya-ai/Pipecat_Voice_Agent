"""Caps how many prior conversation turns get resent to the LLM each call.

Every turn is a fresh stateless Vertex request that resends the full
transcript so far (plus the system prompt and this turn's RAG/price system
messages), so a long call's LLM first-token latency creeps up turn over turn
as that resent context balloons. This trims the tail of the history after
each assistant turn, keeping the original system prompt (index 0) plus only
the most recent MAX_CONTEXT_MESSAGES messages.
"""

import os

from pipecat.frames.frames import Frame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class ContextTrimmer(FrameProcessor):
    MAX_CONTEXT_MESSAGES = int(os.getenv("MAX_CONTEXT_MESSAGES", "20"))

    def __init__(self, context: LLMContext, **kwargs):
        super().__init__(**kwargs)
        self._context = context

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        msgs = self._context._messages
        if len(msgs) > self.MAX_CONTEXT_MESSAGES + 1:
            msgs[:] = [msgs[0]] + msgs[-self.MAX_CONTEXT_MESSAGES:]

        await self.push_frame(frame, direction)
