"""Embedding model and reranker, loaded once on first use (both multilingual, both local)."""
from functools import lru_cache

import torch

EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL, device=_device())


@lru_cache(maxsize=1)
def reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANK_MODEL, device=_device(), max_length=512)


def embed(texts: list[str], batch_size: int = 16) -> list[list[float]]:
    """Normalised vectors, so cosine similarity is a dot product."""
    vectors = embedder().encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                show_progress_bar=len(texts) > 200)
    return [v.tolist() for v in vectors]


def rerank(question: str, passages: list[str]) -> list[float]:
    """Relevance between 0 and 1 for each passage (sigmoid of the cross-encoder logit)."""
    if not passages:
        return []
    scores = reranker().predict([(question, p) for p in passages], batch_size=16,
                                activation_fn=torch.nn.Sigmoid())
    return [float(s) for s in scores]
