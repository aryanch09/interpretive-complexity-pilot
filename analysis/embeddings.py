"""
Neural embedding computation for cross-persona disagreement.

Design: loads all-MiniLM-L6-v2 directly via the `transformers` library rather
than through `sentence_transformers`, because sentence-transformers 5.x deadlocks
on import on machines that have TensorFlow installed (its __init__.py triggers
TensorFlow's abseil mutex before any user code runs). The fix is to block
TensorFlow from loading in this process and use the underlying transformers
library directly — which is what sentence-transformers does internally.

Model: sentence-transformers/all-MiniLM-L6-v2
  384-dimensional output, fast on CPU, no GPU required.
  Downloaded once by transformers' model hub cache (~90 MB).

The TF-IDF fallback that was used in v2 is preserved below (commented out).
It measured vocabulary overlap, not semantic divergence — two agents can agree
completely while using different words, or disagree while sharing vocabulary.
TF-IDF cosine distance is not a valid proxy for reasoning divergence.
"""

import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Singleton model cache ──────────────────────────────────────────────────────

_tokenizer = None
_model = None
_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _load_model():
    global _tokenizer, _model
    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model

    # Block TensorFlow from initializing in this process.
    # TF 2.x uses the abseil C++ library, which acquires a global mutex during
    # init. On macOS with Python 3.13 + PyTorch 2.12, this mutex is never
    # released, hanging the process. Blocking TF here prevents that path.
    # This affects only the current process — TF is not uninstalled.
    if "tensorflow" not in sys.modules:
        sys.modules["tensorflow"] = None  # type: ignore[assignment]

    import torch
    from transformers import AutoModel, AutoTokenizer

    logger.info("Loading embedding model '%s' (first run downloads ~90 MB) ...", _MODEL_NAME)
    _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
    _model = AutoModel.from_pretrained(_MODEL_NAME)
    _model.eval()
    logger.info("Embedding model loaded.")
    return _tokenizer, _model


def _mean_pool(token_embeddings, attention_mask):
    """Mean pooling over token dimension, weighted by attention mask."""
    import torch
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def encode_texts(texts: list[str]) -> np.ndarray:
    """
    Encode a list of strings to L2-normalized 384-d embedding vectors.
    Returns an (n, 384) float32 array.
    """
    import torch

    tokenizer, model = _load_model()
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    with torch.no_grad():
        out = model(**encoded)

    pooled = _mean_pool(out.last_hidden_state, encoded["attention_mask"])
    normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
    return normalized.numpy().astype(np.float32)


# ── Embedding cache ────────────────────────────────────────────────────────────

def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _load_cache(cache_path: Path) -> dict:
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f)


def encode_with_cache(texts: list[str], cache_path: Optional[Path] = None) -> np.ndarray:
    """
    Encode texts, using a file cache keyed by SHA-256 hash of each text.
    Cache stores embeddings as lists (JSON-serializable).
    """
    if cache_path is None:
        return encode_texts(texts)

    cache = _load_cache(cache_path)
    keys = [_text_hash(t) for t in texts]
    missing_indices = [i for i, k in enumerate(keys) if k not in cache]

    if missing_indices:
        missing_texts = [texts[i] for i in missing_indices]
        new_embeddings = encode_texts(missing_texts)
        for idx, emb in zip(missing_indices, new_embeddings):
            cache[keys[idx]] = emb.tolist()
        _save_cache(cache, cache_path)

    embeddings = np.array([cache[k] for k in keys], dtype=np.float32)
    return embeddings


# ── Cosine distance ────────────────────────────────────────────────────────────

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cosine_similarity, clipped to [0, 2]."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 1.0
    return float(np.clip(1.0 - np.dot(a, b) / (norm_a * norm_b), 0.0, 2.0))


# ── Main public function ───────────────────────────────────────────────────────

def compute_embedding_disagreement(
    agent_outputs: list[str],
    cache_path: Optional[Path] = None,
) -> float:
    """
    Mean pairwise cosine distance between neural embeddings of agent reasoning texts.

    Args:
        agent_outputs: list of three reasoning strings (one per persona).
        cache_path: optional Path to embedding_cache.json for caching.

    Returns:
        Mean pairwise cosine distance in [0, 2]. Higher = more disagreement.
        Returns 0.0 if fewer than 2 valid texts are provided.
    """
    valid = [t for t in agent_outputs if t and t.strip()]
    if len(valid) < 2:
        return 0.0

    try:
        embeddings = encode_with_cache(valid, cache_path=cache_path)
        n = len(embeddings)
        distances = [
            _cosine_distance(embeddings[i], embeddings[j])
            for i in range(n)
            for j in range(i + 1, n)
        ]
        return float(np.mean(distances))
    except Exception as exc:
        logger.error("Neural embedding computation failed: %s — returning 0.0", exc)
        return 0.0


def validate_embedding_variation(scores: list[float], threshold: float = 0.01) -> None:
    """
    Raise if all embedding disagreement scores are within `threshold` of each other.
    Called after a full pilot run to catch silent TF-IDF fallback.
    """
    if len(scores) < 2:
        return
    spread = max(scores) - min(scores)
    if spread <= threshold:
        raise RuntimeError(
            f"Embedding measure shows no variation (spread={spread:.4f} ≤ {threshold}) — "
            "neural embeddings may not be loaded correctly. "
            "Check that sys.modules['tensorflow'] = None is set before any transformers import."
        )


# ── TF-IDF baseline (replaced, preserved for reference) ───────────────────────
#
# The v2 pipeline computed embedding_disagreement using TF-IDF cosine distance.
# This measures vocabulary overlap between agent outputs, not semantic divergence.
# It was replaced because: (1) agents routinely use the same financial vocabulary
# regardless of whether they agree or disagree; (2) the pilot result was a flat
# 0.029 spread across all three tiers, which is consistent with vocabulary overlap
# being constant across firm sizes; (3) the advisor's comment that the embedding
# measure "has not been built yet" referred specifically to this limitation.
#
# def _tfidf_disagreement(texts: list[str]) -> float:
#     from sklearn.feature_extraction.text import TfidfVectorizer
#     from sklearn.metrics.pairwise import cosine_distances
#     if len(texts) < 2:
#         return 0.0
#     try:
#         vec = TfidfVectorizer(max_features=500, stop_words="english")
#         mat = vec.fit_transform(texts).toarray()
#         n = len(mat)
#         dists = [cosine_distances(mat[i:i+1], mat[j:j+1])[0][0]
#                  for i in range(n) for j in range(i+1, n)]
#         return float(np.mean(dists))
#     except Exception:
#         return 0.0
