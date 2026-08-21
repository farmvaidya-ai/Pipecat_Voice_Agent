from chonkie import SlumberChunker

from .genie import VertexGeminiGenie

TOKENIZER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Was 2048 until 2026-08-18 — that budget let SlumberChunker merge several
# unrelated FAQ items into one chunk (524-line APS KB -> only 25 chunks for
# ~90 FAQs), which diluted dense-embedding scores for genuinely on-topic
# queries (e.g. "who are your founders" scored only 0.52) while giving
# off-topic queries a higher score against those same oversized,
# topically-broad chunks (a false match scored 0.60) — so even after this
# fix, no fixed dense-score cutoff reliably separates true from false
# matches (see chonkie_rag/search.py's DENSE_TOP_N comment, which dropped
# the old score-threshold gate entirely for that reason). 400 keeps chunks
# close to one-FAQ-per-chunk anyway, which still helps ranking quality even
# though it's no longer load-bearing for a hard cutoff.
CHUNK_SIZE = 400

_chunker: SlumberChunker | None = None


def _get_chunker() -> SlumberChunker:
    global _chunker
    if _chunker is None:
        print(f"Loading SlumberChunker (chunk_size={CHUNK_SIZE})")
        _chunker = SlumberChunker(
            genie=VertexGeminiGenie(),
            tokenizer=TOKENIZER,
            chunk_size=CHUNK_SIZE,
            min_characters_per_chunk=20,
        )
        print("SlumberChunker ready.\n")
    return _chunker


def chunk_text(text: str, source: str) -> list[dict]:
    """Chunk raw KB text with the same SlumberChunker used for the base KB."""
    raw_chunks = _get_chunker().chunk(text)
    chunks = []
    for idx, raw in enumerate(raw_chunks, start=1):
        chunks.append({
            "id": idx,
            "text": raw.text,
            "embed_text": raw.text,
            "token_count": raw.token_count,
            "source": source,
        })
    return chunks
