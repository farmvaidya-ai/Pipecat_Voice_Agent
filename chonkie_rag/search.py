import json
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
from chonkie import QdrantHandshake
from rank_bm25 import BM25Okapi

from .config import QDRANT_URL, QDRANT_API_KEY, RAG_COLLECTION
from .embedding import GoogleVertexEmbeddings, _get_raw_model, embed_query
from . import session_log
from loguru import logger

COLLECTION = RAG_COLLECTION

# Was a hard SCORE_THRESHOLD=0.7 dense-cosine gate until 2026-08-18 — dropped
# because it had no safe value for this embedding model on this corpus: real
# queries confirmed live (chonkie_rag/cache/dense_retrievals.jsonl,
# 2026-08-18) show true and false matches landing in the same score band —
# e.g. "who are the founders of your company" scored 0.52-0.64 against the
# real founders chunk, while an unrelated weather/pincode query scored 0.59-
# 0.61 against a random chunk. No cutoff separates them, so instead of
# gating on a raw number, the top DENSE_TOP_N dense hits are always surfaced
# (regardless of score) and handed to the LLM as plain passages — relevance
# filtering happens there instead, via system_prompt.txt's existing "only
# use the part of the passages that directly answers what the caller asked
# THIS turn" instruction, which reads the actual chunk text rather than a
# similarity number. BM25 keyword hits (a real term-overlap signal, unlike
# raw cosine here) still admit candidates independently, same as before.
DENSE_TOP_N = 2

# Reciprocal Rank Fusion constant (Cormack et al. 2009) — the widely-used
# default; larger values flatten the influence of exact rank position.
RRF_K = 60

# On-disk copies of the two in-memory indices, rewritten every _build_indices()
# call (i.e. every warmup()) — kept as separate files, one per index, purely
# for inspection between runs. The bot itself only ever reads the in-memory
# globals below; these files are not read back on startup.
CACHE_DIR = Path(__file__).parent / "cache"
BM25_CACHE_FILE = CACHE_DIR / "bm25_chunks.json"
DENSE_CACHE_FILE = CACHE_DIR / "dense_vectors.npy"
# .npy is a binary format (raw float32 bytes) — not viewable in a text editor.
# This is a human-readable sidecar: first 8 dims + norm per row, so the
# alignment with bm25_chunks.json can be eyeballed without running numpy.
DENSE_PREVIEW_FILE = CACHE_DIR / "dense_vectors_preview.json"

# Per-query retrieval logs — one line appended per search() call, dense and
# BM25 kept in separate files since they're independent retrieval passes
# (see search()'s docstring) fused only afterward via RRF.
DENSE_RETRIEVAL_LOG = CACHE_DIR / "dense_retrievals.jsonl"
BM25_RETRIEVAL_LOG = CACHE_DIR / "bm25_retrievals.jsonl"

# Plain file appends (not a loguru sink), so they need their own size cap —
# these grew to 260+ MB unrotated before. Single-backup rotation (like
# logrotate's simplest form): once a file crosses this size, the current
# file becomes the one ".1" backup and a fresh one starts.
_RETRIEVAL_LOG_MAX_BYTES = 20 * 1024 * 1024

_handshake: QdrantHandshake | None = None
_bm25_index: BM25Okapi | None = None
_bm25_chunks: list[dict] = []  # payload dicts, same order as _bm25_index's corpus
_bm25_chunk_by_id: dict[int, dict] = {}
_bm25_token_sets: list[frozenset] = []  # tokenized chunk text, same order/index as _bm25_chunks
_dense_vectors: np.ndarray | None = None  # L2-normalized, row i matches _bm25_chunks[i]

# Generic Telugu function/connector words that show up in nearly every chunk
# regardless of topic (per bm25_chunks.json document-frequency counts, most
# of these sit at 70-90% of the corpus). Left un-filtered, a single one of
# these matching between a query and an unrelated chunk is enough to clear
# BM25's OR-gate in search() and inject irrelevant context — e.g. "పేరు"
# ("name") appearing both in "నా పేరు కౌశిక్ అండి" (a caller's self-
# introduction) and, coincidentally, in an unrelated pest-ID chunk's "how
# did it get that name?" aside. Domain terms (పురుగు "pest", ధర "price",
# ట్రాప్స్ "traps") are deliberately not in this list.
_STOPWORDS = frozenset({
    "పేరు", "అంటే", "లో", "మరియు", "ను", "ఒక", "ఈ", "ఏ",
    "వరకు", "నుంచి", "రెండు", "దీనికి", "ఆ", "ఉంటుంది",
})


def _rotate_if_large(path: Path) -> None:
    if path.exists() and path.stat().st_size > _RETRIEVAL_LOG_MAX_BYTES:
        backup = path.with_suffix(path.suffix + ".1")
        backup.unlink(missing_ok=True)
        path.rename(backup)


def _log_retrieval(path: Path, question: str, chunks: list[dict]) -> None:
    """Append one JSON line recording what a single retrieval pass picked."""
    CACHE_DIR.mkdir(exist_ok=True)
    _rotate_if_large(path)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "chunks": chunks,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _tokenize(text: str) -> list[str]:
    """
    Whitespace/punctuation split. Telugu is space-delimited between words,
    so this is enough for BM25's exact-term matching — no stemmer needed.

    Groups by Unicode category (letters/marks/numbers) instead of `\\w+`,
    since `\\w` excludes category M (combining marks) — Telugu and other
    Indic abugidas build syllables out of base consonants + vowel-sign marks,
    so a `\\w+`-based split was shattering every word into bare consonant
    fragments (e.g. "పేరు" -> "ప", "ర"). Those fragments occur in nearly
    every chunk in the corpus regardless of topic, which was making BM25
    treat almost any query as a strong keyword match against almost any
    chunk (see bot_processors/echo.py's near-identical bug).
    """
    tokens: list[str] = []
    current: list[str] = []
    for ch in text:
        if unicodedata.category(ch)[0] in ("L", "M", "N"):
            current.append(ch)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return [t for t in tokens if t not in _STOPWORDS]


def _get_handshake() -> QdrantHandshake:
    """
    Lazy singleton QdrantHandshake — same server/collection as chonkie's vectordb.py.
    """
    global _handshake
    if _handshake is None:
        _handshake = QdrantHandshake(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            collection_name=COLLECTION,
            embedding_model=GoogleVertexEmbeddings(),
            timeout=30,
            # httpx's default keepalive_expiry is 5s — shorter than the gap
            # between most conversational turns, so the pooled HTTPS
            # connection to Qdrant Cloud (us-east-1) was being torn down and
            # re-handshaked (fresh TCP+TLS) on almost every turn. Holding it
            # open for 2 minutes covers normal thinking/speaking pauses
            # without keeping a connection alive indefinitely.
            limits=httpx.Limits(max_keepalive_connections=20, keepalive_expiry=120.0),
        )
        logger.info(f"🔌 Qdrant connected: collection={COLLECTION}")
    return _handshake


def _build_indices() -> None:
    """
    Scroll every point out of the Qdrant collection once (including its
    vector this time) and build two in-memory indices over the same corpus:
    BM25 for exact keyword overlap, and a cached dense-vector matrix for
    cosine similarity.

    The dense vectors are cached here so query-time similarity search never
    has to round-trip to Qdrant (see _local_dense_search) — the collection is
    small enough (tens of chunks) that holding it fully in memory and doing a
    brute-force numpy dot product is sub-millisecond, no ANN index needed.
    Qdrant remains the system of record for ingestion; this cache is only
    refreshed on process restart (i.e. the next warmup() call).
    """
    global _bm25_index, _bm25_chunks, _bm25_chunk_by_id, _bm25_token_sets, _dense_vectors
    client = _get_handshake().client

    chunks: list[dict] = []
    vectors: list[list[float]] = []
    next_offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=COLLECTION,
            with_payload=True,
            with_vectors=True,
            limit=256,
            offset=next_offset,
        )
        for p in points:
            payload = p.payload or {}
            chunks.append({
                "id": payload.get("chunk_seq_id", p.id),
                "text": payload.get("text", payload.get("embed_text", "")),
                "source": payload.get("source", ""),
            })
            vectors.append(p.vector)
        if next_offset is None:
            break

    tokenized_corpus = [_tokenize(c["text"]) for c in chunks]
    _bm25_index = BM25Okapi(tokenized_corpus)
    _bm25_chunks = chunks
    _bm25_chunk_by_id = {c["id"]: c for c in chunks}
    _bm25_token_sets = [frozenset(tokens) for tokens in tokenized_corpus]

    dense_dim = 0
    if vectors:
        vecs = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        _dense_vectors = vecs / norms
        dense_dim = _dense_vectors.shape[1]
    else:
        _dense_vectors = None

    CACHE_DIR.mkdir(exist_ok=True)
    with open(BM25_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    if _dense_vectors is not None:
        np.save(DENSE_CACHE_FILE, _dense_vectors)
        preview = [
            {
                "id": chunks[i]["id"],
                "source": chunks[i]["source"],
                "norm": float(np.linalg.norm(_dense_vectors[i])),
                "vector_preview_first_8_dims": _dense_vectors[i][:8].tolist(),
            }
            for i in range(len(chunks))
        ]
        with open(DENSE_PREVIEW_FILE, "w", encoding="utf-8") as f:
            json.dump(preview, f, ensure_ascii=False, indent=2)

    logger.info(
        f"🔤 BM25 + dense cache built: {len(chunks)} chunks, dense dim={dense_dim} "
        f"(saved to {BM25_CACHE_FILE.name}, {DENSE_CACHE_FILE.name}, {DENSE_PREVIEW_FILE.name})"
    )


def warmup() -> None:
    """
    Force the lazy embedding model(s) + Qdrant connection to load now.

    embed_query() (used by search()) draws from its own module-level
    singleton in embedding.py (_raw_model), separate from the
    GoogleVertexEmbeddings instance handed to QdrantHandshake below — so
    both must be primed here, or the _raw_model one still loads cold on
    the first live call.

    Call this once at process startup (off the asyncio event loop) so the
    cold-start cost doesn't land in the middle of a live call's first RAG
    lookup.
    """
    _get_handshake()
    _get_raw_model()
    _build_indices()


def _local_dense_search(query_vector, candidate_k: int) -> list[dict]:
    """
    Cosine similarity against the in-memory dense-vector cache built in
    _build_indices — replaces the Qdrant network round-trip that used to
    happen here on every query. Same semantics as before (score = cosine
    similarity, same candidate_k), just computed locally.
    """
    if _dense_vectors is None or not _bm25_chunks:
        return []

    q = np.asarray(query_vector, dtype=np.float32)
    q_norm = np.linalg.norm(q)
    if q_norm > 0:
        q = q / q_norm

    scores = _dense_vectors @ q
    k = min(candidate_k, len(scores))
    top_idx = np.argsort(-scores)[:k]

    return [
        {
            "id": _bm25_chunks[i]["id"],
            "score": float(scores[i]),
            "text": _bm25_chunks[i]["text"],
            "source": _bm25_chunks[i]["source"],
        }
        for i in top_idx
    ]


def search(question: str, top_k: int = 6) -> list[dict]:
    """
    Hybrid retrieval: dense (meaning) search + BM25 (exact keyword) search,
    combined via Reciprocal Rank Fusion.

    A chunk is kept as a candidate if EITHER it's in the top DENSE_TOP_N dense
    hits (regardless of raw score — see DENSE_TOP_N's comment for why a score
    cutoff doesn't work here) OR it has a real BM25 keyword overlap. This
    means search() basically never comes back empty for a non-gibberish
    question anymore — that's intentional; the LLM sees the actual passage
    text and decides relevance itself (system_prompt.txt: "only use the part
    of the passages that directly answers what the caller asked THIS turn"),
    which reads the content instead of trusting an unreliable similarity
    number. A genuinely blank/gibberish query still comes back empty since
    there's nothing for dense search to even rank.
    """
    stripped = question.strip()
    candidate_k = top_k

    t0 = time.perf_counter()
    query_vector = embed_query(stripped)
    t1 = time.perf_counter()
    dense_points = _local_dense_search(query_vector, candidate_k)
    t2 = time.perf_counter()
    logger.info(
        f"⏱️ RAG timing: embed={t1 - t0:.3f}s dense_search={t2 - t1:.3f}s"
    )

    dense_rank: dict[int, int] = {}
    dense_score: dict[int, float] = {}
    chunk_by_id: dict[int, dict] = {}
    for rank, p in enumerate(dense_points, start=1):
        cid = p["id"]
        dense_rank[cid] = rank
        dense_score[cid] = p["score"]
        chunk_by_id[cid] = {
            "id": cid,
            "text": p["text"],
            "source": p["source"],
        }
    _log_retrieval(
        DENSE_RETRIEVAL_LOG,
        stripped,
        [
            {"id": p["id"], "score": p["score"], "text": p["text"], "source": p["source"]}
            for p in dense_points
        ],
    )
    logger.info(f"🟣 Dense retrieved: {len(dense_points)} chunks")
    for i, p in enumerate(dense_points, 1):
        snippet = p["text"][:80].replace("\n", " ")
        logger.info(f"  dense {i}: id={p['id']} score={p['score']:.4f} source={p['source']!r} text={snippet!r}...")

    bm25_rank: dict[int, int] = {}
    bm25_score: dict[int, float] = {}
    bm25_hits: list[dict] = []
    if _bm25_index is not None and _bm25_chunks:
        query_tokens = _tokenize(stripped)
        raw_scores = _bm25_index.get_scores(query_tokens)
        query_token_set = frozenset(query_tokens)
        # A single incidental keyword overlap (score > 0) was enough to admit
        # a chunk as a BM25 candidate — e.g. one rare content word shared
        # between an otherwise-unrelated chunk and the query. Require at
        # least 2 distinct overlapping (non-stopword) terms to count as a
        # real keyword match, unless the query itself only has 1 content
        # word to begin with (short queries like "ధర ఎంత?" shouldn't be
        # penalized for not having a second word to overlap on).
        min_overlap = 1 if len(query_token_set) <= 1 else 2

        # Every chunk with a nonzero BM25 score, regardless of overlap count —
        # what search() used to select from before the min_overlap gate was
        # added. Not used for selection, only logged below so it's possible to
        # see what the stricter gate is excluding on a given query.
        pre_gate_scored = [
            (chunk["id"], score)
            for chunk, score in zip(_bm25_chunks, raw_scores)
            if score > 0
        ]
        pre_gate_scored.sort(key=lambda pair: pair[1], reverse=True)

        scored = [
            (chunk["id"], score)
            for chunk, score, token_set in zip(_bm25_chunks, raw_scores, _bm25_token_sets)
            if score > 0 and len(query_token_set & token_set) >= min_overlap
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        for rank, (cid, score) in enumerate(scored[:candidate_k], start=1):
            bm25_rank[cid] = rank
            bm25_score[cid] = score
            if cid not in chunk_by_id:
                chunk_by_id[cid] = _bm25_chunk_by_id[cid]
            chunk = _bm25_chunk_by_id[cid]
            bm25_hits.append({"id": cid, "score": float(score), "text": chunk["text"], "source": chunk["source"]})

        logger.debug(
            f"🟠 BM25 pre-overlap-filter candidates: {len(pre_gate_scored)} "
            f"(min_overlap={min_overlap} gate kept {len(scored)} of these)"
        )
        for i, (cid, score) in enumerate(pre_gate_scored[:candidate_k], 1):
            chunk = _bm25_chunk_by_id[cid]
            snippet = chunk["text"][:80].replace("\n", " ")
            kept = "kept" if cid in bm25_rank else "filtered out"
            logger.debug(
                f"  bm25-raw {i}: id={cid} score={score:.4f} [{kept}] "
                f"source={chunk['source']!r} text={snippet!r}..."
            )
    _log_retrieval(BM25_RETRIEVAL_LOG, stripped, bm25_hits)
    logger.info(f"🟠 BM25 retrieved: {len(bm25_hits)} chunks")
    for i, h in enumerate(bm25_hits, 1):
        snippet = h["text"][:80].replace("\n", " ")
        logger.info(f"  bm25 {i}: id={h['id']} score={h['score']:.4f} source={h['source']!r} text={snippet!r}...")

    candidate_ids = {
        cid for cid in set(dense_rank) | set(bm25_rank)
        if dense_rank.get(cid, 999) <= DENSE_TOP_N or cid in bm25_rank
    }

    # A chunk only counts as dense-confirmed for the overlap check below if
    # it's actually in the top DENSE_TOP_N dense hits — otherwise a chunk
    # admitted purely on a weak one-word BM25 match that also happened to
    # land somewhere further down the (unfiltered) dense top-k would get
    # treated as "Dense+BM25 fused", the strongest selection tier, despite
    # dense search barely ranking it at all. This was the root cause of
    # low-relevance chunks (e.g. a single incidental keyword overlap)
    # outranking genuinely relevant ones.
    dense_confirmed = {cid for cid in dense_rank if dense_rank[cid] <= DENSE_TOP_N}

    fused = []
    for cid in candidate_ids:
        rrf_score = 0.0
        if cid in dense_rank:
            rrf_score += 1.0 / (RRF_K + dense_rank[cid])
        if cid in bm25_rank:
            rrf_score += 1.0 / (RRF_K + bm25_rank[cid])
        fused.append((cid, rrf_score))
    fused.sort(key=lambda pair: pair[1], reverse=True)

    # A chunk counts as truly "fused" only if BOTH retrievers independently
    # surfaced it (retriever == "Dense+BM25"), not just one of them clearing
    # the OR-gate above. When any such double-hit chunk exists, trust those
    # over single-retriever hits and drop the rest; when none exists (the
    # common case for this KB), fall back to the top 3 single-retriever hits
    # instead of the usual top_k, since there's no cross-retriever agreement
    # to lean on.
    overlap_ids = {cid for cid in candidate_ids if cid in dense_confirmed and cid in bm25_rank}
    if overlap_ids:
        selected = [pair for pair in fused if pair[0] in overlap_ids][:top_k]
        selection_mode = "fused-overlap"
    else:
        selected = fused[:3]
        selection_mode = "top3-fallback"

    results = [
        {
            "id": cid,
            "score": rrf_score,
            "text": chunk_by_id[cid]["text"],
            "source": chunk_by_id[cid]["source"],
            "dense_score": dense_score.get(cid),
            "bm25_score": bm25_score.get(cid),
            "retriever": "+".join(
                filter(None, [cid in dense_rank and "Dense", cid in bm25_rank and "BM25"])
            ),
        }
        for cid, rrf_score in selected
    ]

    if results:
        if selection_mode == "fused-overlap":
            logger.info(
                f"✅ RRF fusion: {len(results)} fused (Dense+BM25) chunk(s) kept "
                f"(dense_candidates={len(dense_rank)}, bm25_candidates={len(bm25_rank)}, k={RRF_K})"
            )
        else:
            logger.info(
                f"⚠️ RRF fusion: no Dense+BM25 overlap — falling back to top {len(results)} "
                f"combined (dense_candidates={len(dense_rank)}, bm25_candidates={len(bm25_rank)}, k={RRF_K})"
            )
        for i, r in enumerate(results, 1):
            logger.info(
                f"  #{i} id={r['id']} rrf_score={r['score']:.5f} retriever={r['retriever']} "
                f"dense={r['dense_score']} bm25={r['bm25_score']} source={r['source']!r} "
                f"text={r['text']!r}"
            )
    else:
        logger.info("🔍 RAG hybrid: no chunks above threshold / no keyword overlap")

    return results
