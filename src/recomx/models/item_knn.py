"""Sparse item-neighborhood collaborative filtering for RecomX."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from recomx.models.sparse_interactions import SparseInteractionMatrix


@dataclass(frozen=True)
class ItemKNNConfig:
    """Scale-safe Item-KNN configuration."""

    neighbors: int = 100
    min_similarity: float = 0.0

    def validate(self) -> None:
        if self.neighbors <= 0:
            raise ValueError("neighbors must be positive.")
        if not 0.0 <= self.min_similarity <= 1.0:
            raise ValueError("min_similarity must lie in [0, 1].")


@dataclass(frozen=True)
class ItemNeighborhoods:
    """Bounded sparse item-to-item neighborhood graph."""

    matrix: sparse.csr_matrix
    item_ids: np.ndarray

    @property
    def n_items(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.matrix.nnz)

    def neighbors_for_item(self, item_id: int) -> tuple[np.ndarray, np.ndarray]:
        position = int(np.searchsorted(self.item_ids, item_id))
        if position >= len(self.item_ids) or int(self.item_ids[position]) != int(item_id):
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        start, end = self.matrix.indptr[position], self.matrix.indptr[position + 1]
        columns = self.matrix.indices[start:end]
        return self.item_ids[columns].copy(), self.matrix.data[start:end].copy()


def _top_neighbors(
    indices: np.ndarray,
    scores: np.ndarray,
    item_ids: np.ndarray,
    limit: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic highest-scoring neighbors."""

    if len(scores) == 0:
        return indices, scores

    order = np.lexsort((item_ids[indices], -scores))
    chosen = order[:limit]
    return indices[chosen], scores[chosen]


def build_item_neighborhoods(
    interactions: SparseInteractionMatrix,
    config: ItemKNNConfig | None = None,
) -> ItemNeighborhoods:
    """Build bounded cosine Item-KNN without a dense item-item matrix."""

    active = config or ItemKNNConfig()
    active.validate()

    item_user = interactions.matrix.T.tocsr().astype(np.float32)
    norms = np.sqrt(np.asarray(item_user.multiply(item_user).sum(axis=1)).ravel())
    inverse = np.zeros_like(norms, dtype=np.float32)
    nonzero = norms > 0
    inverse[nonzero] = 1.0 / norms[nonzero]
    normalized = sparse.diags(inverse) @ item_user

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []

    # One sparse row at a time avoids materializing the full item-item matrix.
    for item_index in range(interactions.n_items):
        similarities = (normalized[item_index] @ normalized.T).tocsr()
        candidate_indices = similarities.indices
        candidate_scores = similarities.data.astype(np.float32, copy=False)

        keep = candidate_indices != item_index
        if active.min_similarity > 0:
            keep &= candidate_scores >= active.min_similarity
        else:
            keep &= candidate_scores > 0

        candidate_indices = candidate_indices[keep]
        candidate_scores = candidate_scores[keep]
        candidate_indices, candidate_scores = _top_neighbors(
            candidate_indices,
            candidate_scores,
            interactions.item_ids,
            active.neighbors,
        )

        rows.extend([item_index] * len(candidate_indices))
        columns.extend(candidate_indices.tolist())
        values.extend(candidate_scores.tolist())

    graph = sparse.csr_matrix(
        (
            np.asarray(values, dtype=np.float32),
            (np.asarray(rows, dtype=np.int32), np.asarray(columns, dtype=np.int32)),
        ),
        shape=(interactions.n_items, interactions.n_items),
        dtype=np.float32,
    )
    graph.sort_indices()
    return ItemNeighborhoods(matrix=graph, item_ids=interactions.item_ids.copy())


def recommend_item_knn(
    user_id: int,
    interactions: SparseInteractionMatrix,
    neighborhoods: ItemNeighborhoods,
    top_k: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Recommend unseen items by aggregating similarities from positive history."""

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    user_index = interactions.user_index(user_id)
    if user_index is None:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    start = interactions.matrix.indptr[user_index]
    end = interactions.matrix.indptr[user_index + 1]
    history = interactions.matrix.indices[start:end]
    if len(history) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    scores = np.asarray(neighborhoods.matrix[history].sum(axis=0)).ravel()
    scores[history] = 0.0
    candidates = np.flatnonzero(scores > 0)
    if len(candidates) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    candidate_scores = scores[candidates].astype(np.float32, copy=False)
    order = np.lexsort((interactions.item_ids[candidates], -candidate_scores))
    chosen = order[:top_k]
    selected = candidates[chosen]
    return interactions.item_ids[selected].copy(), candidate_scores[chosen].copy()
