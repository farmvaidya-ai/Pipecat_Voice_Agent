from chonkie import SlumberChunker

from .genie import VertexGeminiGenie

TOKENIZER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 2048

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
