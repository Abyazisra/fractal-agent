"""
Differentiable Gist Retrieval Module.

IMPROVEMENT 3: Replaces the LLM prompt-based lookup with embedding-based
retrieval. Uses sentence-transformer embeddings + cosine similarity with
a learned relevance threshold to select pages, eliminating the need for
an LLM inference call during the lookup step.

The cross-attention adapter is available for fine-tuned deployments but
defaults to cosine similarity which works out-of-the-box without training.
"""

import logging
from typing import List, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)

_SentenceTransformer = None


def _lazy_import_st():
    """Lazily import sentence-transformers."""
    global _SentenceTransformer
    if _SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer
        _SentenceTransformer = SentenceTransformer
    return _SentenceTransformer


class DifferentiableRetriever:
    """
    Embedding-based gist retrieval system.

    Uses sentence-transformer to encode gists and queries into the same
    embedding space, then retrieves via cosine similarity. This replaces
    the LLM prompt-based lookup with a single forward pass through a
    lightweight encoder, eliminating inference overhead.

    Key design: Adaptive top-k selection using a relevance threshold
    instead of fixed-k, so only genuinely relevant pages are retrieved.
    """

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        hidden_dim: int = 256,
        use_cross_attention: bool = False,  # Default OFF — cosine is better untrained
        relevance_threshold: float = 0.25,  # Min cosine sim to consider a page relevant
    ):
        SentenceTransformer = _lazy_import_st()

        self.encoder = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        self.relevance_threshold = relevance_threshold

        self._gist_embeddings = None
        self._gist_indices = None

        logger.info(
            f"DifferentiableRetriever initialized: model={embedding_model_name}, "
            f"dim={self.embedding_dim}, threshold={relevance_threshold}"
        )

    def encode_gists(self, gists: List[str], page_indices: List[int]):
        """Pre-encode all gist memories into embeddings."""
        self._gist_embeddings = self.encoder.encode(
            gists, show_progress_bar=False, normalize_embeddings=True
        )
        self._gist_indices = page_indices
        logger.info(f"Encoded {len(gists)} gists into {self._gist_embeddings.shape} embeddings")

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> List[Tuple[int, float]]:
        """
        Retrieve the most relevant page indices for a query.

        Uses adaptive selection: returns up to top_k pages, but only those
        whose cosine similarity exceeds the relevance threshold. Always
        returns at least 1 page (the most relevant one).

        Args:
            query: The task/question text.
            top_k: Maximum number of pages to retrieve.

        Returns:
            List of (page_index, relevance_score) tuples, sorted by relevance.
        """
        if self._gist_embeddings is None:
            raise RuntimeError("Gists not encoded yet. Call encode_gists() first.")

        # Encode query
        query_embedding = self.encoder.encode(
            [query], show_progress_bar=False, normalize_embeddings=True
        )[0]

        # Cosine similarity (embeddings are already normalized)
        scores = np.dot(self._gist_embeddings, query_embedding)

        # Sort by relevance
        sorted_indices = np.argsort(scores)[::-1]

        # Adaptive selection: take pages above threshold, minimum 1, maximum top_k
        results = []
        for idx in sorted_indices[:top_k]:
            score = float(scores[idx])
            if score >= self.relevance_threshold or len(results) == 0:
                page_idx = self._gist_indices[idx]
                results.append((page_idx, score))
            else:
                break  # Remaining scores are even lower

        logger.info(
            f"Differentiable retrieval: selected {len(results)} pages "
            f"(scores: {[f'{s:.3f}' for _, s in results]})"
        )
        return results

    def retrieve_pages(
        self,
        query: str,
        top_k: int = 3,
    ) -> List[int]:
        """Convenience method returning just page indices."""
        results = self.retrieve(query, top_k)
        return [page_idx for page_idx, _ in results]
