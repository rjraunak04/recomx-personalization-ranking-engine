"""Train-only latent-factor retrieval using sparse Truncated SVD."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import svds

from recomx.models.sparse_interactions import SparseInteractionMatrix


@dataclass(frozen=True)
class LatentSVDConfig:
    """Configuration for the deterministic sparse latent-factor model."""

    factors: int = 64
    random_state: int = 42

    def validate(self, shape: tuple[int, int] | None = None) -> None:
        if self.factors <= 0:
            raise ValueError("factors must be positive.")
        if self.random_state < 0:
            raise ValueError("random_state must be non-negative.")
        if shape is not None and self.factors >= min(shape):
            raise ValueError("factors must be smaller than both matrix dimensions.")


@dataclass(frozen=True)
class LatentSVDModel:
    """Compact user/item embeddings reconstructed from the train-only CSR matrix."""

    user_factors: np.ndarray
    item_factors: np.ndarray
    user_ids: np.ndarray
    item_ids: np.ndarray

    @property
    def n_factors(self) -> int:
        return int(self.user_factors.shape[1])

    def recommend(
        self,
        user_id: int,
        interactions: SparseInteractionMatrix,
        top_k: int = 20,
        batch_size: int = 4096,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Rank unseen items for one warm user without dense user-item reconstruction."""

        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        user_index = interactions.user_index(user_id)
        if user_index is None:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        start = interactions.matrix.indptr[user_index]
        end = interactions.matrix.indptr[user_index + 1]
        seen = set(interactions.matrix.indices[start:end].tolist())

        user_vector = self.user_factors[user_index]
        best_indices = np.array([], dtype=np.int64)
        best_scores = np.array([], dtype=np.float32)

        for offset in range(0, len(self.item_ids), batch_size):
            stop = min(offset + batch_size, len(self.item_ids))
            scores = self.item_factors[offset:stop] @ user_vector
            indices = np.arange(offset, stop, dtype=np.int64)
            if seen:
                mask = np.fromiter((int(i) not in seen for i in indices), dtype=bool)
                indices = indices[mask]
                scores = scores[mask]
            if len(indices) == 0:
                continue
            merged_indices = np.concatenate((best_indices, indices))
            merged_scores = np.concatenate((best_scores, scores.astype(np.float32)))
            order = np.lexsort((self.item_ids[merged_indices], -merged_scores))
            chosen = order[:top_k]
            best_indices = merged_indices[chosen]
            best_scores = merged_scores[chosen]

        return self.item_ids[best_indices].copy(), best_scores.copy()


def fit_latent_svd(
    interactions: SparseInteractionMatrix,
    config: LatentSVDConfig | None = None,
) -> LatentSVDModel:
    """Fit sparse SVD and return balanced user/item latent factors."""

    active = config or LatentSVDConfig()
    active.validate(interactions.matrix.shape)

    rng = np.random.default_rng(active.random_state)
    start_vector = rng.standard_normal(min(interactions.matrix.shape)).astype(np.float32)

    user_vectors, singular_values, item_vectors_t = svds(
        interactions.matrix,
        k=active.factors,
        v0=start_vector,
        which="LM",
        return_singular_vectors=True,
    )
    order = np.argsort(singular_values)[::-1]
    singular_values = singular_values[order].astype(np.float32)
    user_vectors = user_vectors[:, order].astype(np.float32)
    item_vectors_t = item_vectors_t[order, :].astype(np.float32)

    scale = np.sqrt(singular_values).astype(np.float32)
    user_factors = np.ascontiguousarray(user_vectors * scale)
    item_factors = np.ascontiguousarray(item_vectors_t.T * scale)

    return LatentSVDModel(
        user_factors=user_factors,
        item_factors=item_factors,
        user_ids=interactions.user_ids.copy(),
        item_ids=interactions.item_ids.copy(),
    )
