"""Word-batch text aggregator for lower time-to-first-audio in TTS.

Pipecat's default SimpleTextAggregator buffers LLM output until a full
sentence is ready before handing text to the TTS service. This aggregator
instead flushes every `word_batch_size` words, extended to the next clause
boundary (comma, semicolon, colon, em dash) if one falls within
`clause_lookahead_words` words — so TTS synthesis starts mid-sentence
instead of waiting for the whole thing. Sentence-ending punctuation always
flushes immediately regardless of word count, so short utterances aren't
held up waiting for a threshold that may never be reached.
"""

import re
from collections.abc import AsyncIterator

from pipecat.utils.string import SENTENCE_ENDING_PUNCTUATION
from pipecat.utils.text.base_text_aggregator import Aggregation, AggregationType, BaseTextAggregator

_CLAUSE_PUNCTUATION = {",", ";", ":", "—","."}


class WordBatchTextAggregator(BaseTextAggregator):
    """Flushes text every N words (or sooner, at sentence/clause boundaries)."""

    def __init__(
        self,
        word_batch_size: int = 7,
        clause_lookahead_words: int = 3,
        **kwargs,
    ):
        kwargs.setdefault("aggregation_type", AggregationType.WORD)
        super().__init__(**kwargs)
        self._text = ""
        self._word_batch_size = word_batch_size
        self._clause_lookahead_words = clause_lookahead_words

    @property
    def text(self) -> Aggregation:
        return Aggregation(text=self._text.strip(), type=self._aggregation_type)

    async def aggregate(self, text: str) -> AsyncIterator[Aggregation]:
        self._text += text

        while True:
            result = self._try_extract_chunk()
            if result is None:
                break
            yield result

    def _try_extract_chunk(self) -> Aggregation | None:
        # Whichever boundary comes first in the buffer wins — a distant
        # sentence end must not swallow an earlier word-count cutoff (e.g.
        # when a large chunk of text arrives in one delta).
        sentence_cut = self._find_sentence_cut()

        words = self._complete_words()
        if len(words) < self._word_batch_size:
            return self._flush_through(sentence_cut) if sentence_cut is not None else None

        word_cut = self._find_word_batch_cut(words)

        if sentence_cut is not None and sentence_cut < word_cut:
            return self._flush_through(sentence_cut)
        return self._flush_through(word_cut)

    def _complete_words(self) -> list[re.Match]:
        # LLMs stream sub-word deltas, not whole words — so the \S+ match
        # touching the very end of the buffer might still be a partial word
        # with more characters arriving in the next delta. Counting it as a
        # finished word let the batch-size cutoff land mid-word (e.g.
        # "ఎర్రముక్కు" arriving as "...ఎర్" then "రముక్కు..." got flushed as
        # two separate TTS calls, each mispronounced on its own). Excluding
        # it here just delays that word's contribution to the count/cut by
        # one delta, until whitespace after it actually confirms it's done.
        words = list(re.finditer(r"\S+", self._text))
        if words and words[-1].end() == len(self._text):
            words = words[:-1]
        return words

    def _find_sentence_cut(self) -> int | None:
        # Requires a following character to confirm (mirrors pipecat's own
        # lookahead: "3." shouldn't flush until we see whether "5" follows).
        for i, char in enumerate(self._text[:-1]):
            if char in SENTENCE_ENDING_PUNCTUATION and self._text[i + 1].isspace():
                if not any(c.isalnum() for c in self._text[: i + 1]):
                    # Nothing but punctuation before this mark — the real
                    # sentence it ends already got flushed by an earlier
                    # word-batch cut, and this mark just arrived on its own
                    # in a later delta (e.g. buffer is now ". తదుపరి..."). Cutting
                    # here would flush a bare "." as its own TTS request,
                    # which Bakbak rejects (input shorter than its vocoder's
                    # kernel size — see bakbak_tts.py). Keep scanning for a
                    # boundary that actually has words before it.
                    continue
                return i
        return None

    def _find_word_batch_cut(self, words: list[re.Match]) -> int:
        search_end = min(len(words), self._word_batch_size + self._clause_lookahead_words)
        for w in words[self._word_batch_size - 1 : search_end]:
            if self._text[w.end() - 1] in _CLAUSE_PUNCTUATION:
                return w.end() - 1
        return words[self._word_batch_size - 1].end() - 1

    def _flush_through(self, index: int) -> Aggregation:
        result = self._text[: index + 1]
        self._text = self._text[index + 1 :]
        return Aggregation(text=result.strip(), type=self._aggregation_type)

    async def flush(self) -> Aggregation | None:
        # A trailing punctuation-only fragment (e.g. a lone "." that arrived
        # as its own delta after the preceding word already hit the batch
        # cutoff) has no phonetic content — sending it to TTS as its own
        # request fails on some backends (e.g. Bakbak rejects input shorter
        # than its vocoder's kernel size). Drop it instead of yielding it.
        stripped = self._text.strip()
        self._text = ""
        if stripped and any(c.isalnum() for c in stripped):
            return Aggregation(text=stripped, type=self._aggregation_type)
        return None

    async def handle_interruption(self):
        self._text = ""

    async def reset(self):
        self._text = ""
