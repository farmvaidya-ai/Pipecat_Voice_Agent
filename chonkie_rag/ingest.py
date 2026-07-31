"""
Command-line KB ingestion — add a new knowledge-base text file to a Qdrant
collection, without touching or recreating any existing content in it (safe
to run alongside an already-populated collection). Each KB (APS, KVK, ...)
lives in its own separate collection — pass which one to target.

Usage (from the pipecat project root, with its venv active):
    python -m chonkie_rag.ingest "path/to/new_kb.txt" "source_label" [collection_name]

If collection_name is omitted, it defaults to RAG_COLLECTION from .env (the
collection Bot.py is currently configured to search) — so if you're only
ever adding to the one KB you're actively using, you can leave it off.

After it finishes, restart Bot.py so chonkie_rag/search.py's in-memory
BM25/dense cache (built once at warmup()) picks up the new chunks.
"""

import sys

from chonkie import QdrantHandshake
from qdrant_client.models import PointStruct

from .config import QDRANT_URL, QDRANT_API_KEY, RAG_COLLECTION
from .chunker import chunk_text
from .embedding import GoogleVertexEmbeddings, embed_passages


def _get_handshake(collection: str) -> QdrantHandshake:
    return QdrantHandshake(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=collection,
        embedding_model=GoogleVertexEmbeddings(),
        timeout=60,
    )


def ingest_file(file_path: str, source: str, collection: str = RAG_COLLECTION) -> None:
    with open(file_path, encoding="utf-8-sig") as f:
        text = f.read()
    print(f"Loaded {len(text)} characters from {file_path}")

    chunks = chunk_text(text, source=source)
    print(f"Chunked into {len(chunks)} pieces (source={source!r})")

    texts = [c["embed_text"] for c in chunks]
    vectors = embed_passages(texts)

    hs = _get_handshake(collection)
    # Offset new chunk ids past whatever's already in the collection so this
    # can never collide with / overwrite existing points from another source.
    existing = hs.client.count(collection_name=collection).count
    points = [
        PointStruct(
            id=existing + chunk["id"],
            vector=vector,
            payload={
                "text": chunk["text"],
                "embed_text": chunk["embed_text"],
                "token_count": chunk["token_count"],
                "source": chunk["source"],
                "chunk_seq_id": existing + chunk["id"],
            },
        )
        for chunk, vector in zip(chunks, vectors)
    ]
    # Plain upsert() sends every point in one HTTP request — fine for small
    # batches, but a few hundred embedding vectors in one write reliably hit
    # WriteTimeout on this connection. upload_points() splits the same points
    # into smaller batches under the hood (with retry), so no single request
    # is big enough to time out.
    hs.client.upload_points(collection_name=collection, points=points, batch_size=64, wait=True)
    print(
        f"Indexed {len(points)} chunks into '{collection}' "
        f"(ids {existing + 1}..{existing + len(points)})"
    )
    print("\nDone. Restart Bot.py so its in-memory RAG cache picks up the new content.")


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print(
            'Usage: python -m chonkie_rag.ingest "path/to/file.txt" "source_label" [collection_name]'
        )
        sys.exit(1)
    target_collection = sys.argv[3] if len(sys.argv) == 4 else RAG_COLLECTION
    ingest_file(sys.argv[1], sys.argv[2], target_collection)
