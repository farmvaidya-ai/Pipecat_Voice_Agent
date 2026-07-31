import numpy as np
import vertexai
from google.api_core.exceptions import InvalidArgument
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from tokenizers import Tokenizer
from chonkie.embeddings.base import BaseEmbeddings

from .config import PROJECT_ID, LOCATION
from . import session_log
from loguru import logger

MODEL_NAME = "text-multilingual-embedding-002"
VECTOR_DIM = 768

vertexai.init(project=PROJECT_ID, location=LOCATION)


def _embed_with_retry(model: TextEmbeddingModel, batch: list) -> list:
    """
    Call get_embeddings, halving the batch and retrying on a token-limit error.

    MiniLM token counts are only a rough proxy for Vertex's own tokenizer, and
    SlumberChunker's LLM-guided boundaries produce very unevenly sized chunks
    (some far larger than the configured chunk_size), so a fixed MiniLM->Vertex
    conversion factor isn't reliable for pre-sizing batches. This makes batching
    self-correcting instead of depending on that guess.
    """
    try:
        return model.get_embeddings(batch)
    except InvalidArgument:
        if len(batch) == 1:
            raise
        mid = len(batch) // 2
        return _embed_with_retry(model, batch[:mid]) + _embed_with_retry(model, batch[mid:])


class GoogleVertexEmbeddings(BaseEmbeddings):
    """
    Wraps Vertex AI text-multilingual-embedding-002 as a chonkie BaseEmbeddings.

    API limits for this model:
      - Max 250 instances per request
      - Max 20 000 tokens per request (Vertex tokenizer)

    The MiniLM token count is used as a rough pre-batching heuristic to keep
    the number of API calls down; _embed_with_retry is the real safety net
    since the MiniLM->Vertex conversion ratio varies a lot per chunk.
    """

    _MAX_INSTANCES = 250
    _MAX_TOKENS = 1200

    def __init__(self) -> None:
        super().__init__()
        print(f"Loading {MODEL_NAME}...")
        self._model = TextEmbeddingModel.from_pretrained(MODEL_NAME)
        # MiniLM tokenizer is used only for token-budget counting.
        self._tokenizer = Tokenizer.from_pretrained(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        print(f"{MODEL_NAME} ready. Dimension: {VECTOR_DIM}\n")
        logger.info(f"📚 Embedding model ready: {MODEL_NAME} (dim={VECTOR_DIM})")

    # ── Internal helpers ───────────────────────────────────────────────────

    def _token_count(self, text: str) -> int:
        return len(self._tokenizer.encode(text).ids)

    def _call_api(self, inputs: list[str | TextEmbeddingInput]) -> list:
        """Send one API call, splitting the batch if it exceeds Vertex's token limit."""
        return _embed_with_retry(self._model, inputs)

    # ── BaseEmbeddings interface ───────────────────────────────────────────

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text with SEMANTIC_SIMILARITY task type."""
        result = self._model.get_embeddings(
            [TextEmbeddingInput(text, "SEMANTIC_SIMILARITY")]
        )
        return np.array(result[0].values, dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """
        Embed texts in safe sub-batches respecting both API limits:
          - _MAX_INSTANCES per call
          - _MAX_TOKENS cumulative MiniLM token count per call
        """
        all_results = []
        batch: list[str | TextEmbeddingInput] = []
        batch_tokens = 0

        for text in texts:
            t = self._token_count(text)
            if batch and (
                len(batch) >= self._MAX_INSTANCES
                or batch_tokens + t > self._MAX_TOKENS
            ):
                all_results.extend(self._call_api(batch))
                batch = []
                batch_tokens = 0
            batch.append(TextEmbeddingInput(text, "SEMANTIC_SIMILARITY"))
            batch_tokens += t

        if batch:
            all_results.extend(self._call_api(batch))

        return [np.array(r.values, dtype=np.float32) for r in all_results]

    @property
    def dimension(self) -> int:
        return VECTOR_DIM

    def get_tokenizer(self):
        return self._tokenizer


# ── Task-specific helpers (for RAG pipelines) ─────────────────────────────
# Singleton raw model shared by embed_passages / embed_query.

_raw_model: TextEmbeddingModel | None = None


def _get_raw_model() -> TextEmbeddingModel:
    global _raw_model
    if _raw_model is None:
        print(f"Loading {MODEL_NAME}...")
        _raw_model = TextEmbeddingModel.from_pretrained(MODEL_NAME)
        print("Model ready.\n")
    return _raw_model


def embed_query(text: str) -> list[float]:
    """Embed a user query for retrieval (RETRIEVAL_QUERY task)."""
    model = _get_raw_model()
    result = model.get_embeddings([TextEmbeddingInput(text, "RETRIEVAL_QUERY")])
    return result[0].values


def embed_passages(texts: list[str]) -> list[list[float]]:
    """
    Embed document chunks for indexing (RETRIEVAL_DOCUMENT task).

    Applies the same dual-constraint batching as GoogleVertexEmbeddings.embed_batch:
      - max 250 instances per call
      - max 1200 MiniLM-token budget per call, as a rough pre-batching heuristic
        (the real safety net is _embed_with_retry, since MiniLM undercounts
        Vertex's own tokenizer by a widely varying amount per chunk)
    """
    model = _get_raw_model()
    tokenizer = Tokenizer.from_pretrained(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    _MAX_INST = 250
    _MAX_TOK = 1200

    all_vectors: list[list[float]] = []
    batch: list[str | TextEmbeddingInput] = []
    batch_tokens = 0
    batch_num = 1

    for text in texts:
        t = len(tokenizer.encode(text).ids)
        if batch and (len(batch) >= _MAX_INST or batch_tokens + t > _MAX_TOK):
            results = _embed_with_retry(model, batch)
            all_vectors.extend([r.values for r in results])
            print(f"  Embedded batch {batch_num} ({len(batch)} passages)")
            batch_num += 1
            batch = []
            batch_tokens = 0
        batch.append(TextEmbeddingInput(text, "RETRIEVAL_DOCUMENT"))
        batch_tokens += t

    if batch:
        results = _embed_with_retry(model, batch)
        all_vectors.extend([r.values for r in results])
        print(f"  Embedded batch {batch_num} ({len(batch)} passages)")

    return all_vectors
