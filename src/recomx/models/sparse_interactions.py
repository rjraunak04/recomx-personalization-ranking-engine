"""Sparse positive-feedback representation for collaborative recommendation."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import sparse
from recomx.evaluation.relevance import RelevancePolicy

@dataclass(frozen=True)
class SparseInteractionMatrix:
    matrix: sparse.csr_matrix
    user_ids: np.ndarray
    item_ids: np.ndarray
    user_to_index: dict[int, int]
    item_to_index: dict[int, int]

    @property
    def n_users(self) -> int: return int(self.matrix.shape[0])
    @property
    def n_items(self) -> int: return int(self.matrix.shape[1])
    @property
    def n_interactions(self) -> int: return int(self.matrix.nnz)
    @property
    def density(self) -> float:
        possible = self.n_users * self.n_items
        return 0.0 if possible == 0 else self.n_interactions / possible

    def user_index(self, user_id: int) -> int | None:
        return self.user_to_index.get(int(user_id))

    def item_index(self, item_id: int) -> int | None:
        return self.item_to_index.get(int(item_id))

    def positive_items_for_user(self, user_id: int) -> np.ndarray:
        index = self.user_index(user_id)
        if index is None:
            return np.array([], dtype=np.int64)
        start, end = self.matrix.indptr[index], self.matrix.indptr[index + 1]
        return self.item_ids[self.matrix.indices[start:end]].copy()

def build_sparse_interaction_matrix(train: pd.DataFrame, policy: RelevancePolicy | None = None) -> SparseInteractionMatrix:
    required = {"userId", "movieId", "rating"}
    missing = required.difference(train.columns)
    if missing: raise ValueError(f"train missing required columns: {sorted(missing)}")
    if train.empty: raise ValueError("train cannot be empty.")
    active_policy = policy or RelevancePolicy()
    active_policy.validate()
    relevant = active_policy.relevant_interactions(train)
    if relevant.empty: raise ValueError("No relevant training interactions available.")
    pairs = relevant[["userId", "movieId"]].drop_duplicates().astype({"userId": "int64", "movieId": "int64"})
    user_ids = np.sort(pairs["userId"].unique()).astype(np.int64, copy=False)
    item_ids = np.sort(pairs["movieId"].unique()).astype(np.int64, copy=False)
    user_to_index = {int(v): i for i, v in enumerate(user_ids)}
    item_to_index = {int(v): i for i, v in enumerate(item_ids)}
    rows = pairs["userId"].map(user_to_index).to_numpy(dtype=np.int64)
    cols = pairs["movieId"].map(item_to_index).to_numpy(dtype=np.int64)
    matrix = sparse.csr_matrix((np.ones(len(pairs), dtype=np.float32), (rows, cols)), shape=(len(user_ids), len(item_ids)), dtype=np.float32)
    matrix.sum_duplicates(); matrix.sort_indices()
    if matrix.nnz != len(pairs): raise ValueError("Sparse matrix lost unique positive user-item interactions.")
    if matrix.dtype != np.float32: raise ValueError("Sparse interaction matrix must use float32.")
    if matrix.nnz and not np.all(matrix.data == np.float32(1.0)): raise ValueError("Sparse interaction values must be binary 1.0.")
    return SparseInteractionMatrix(matrix, user_ids, item_ids, user_to_index, item_to_index)
