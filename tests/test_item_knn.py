"""Tests for sparse Item-KNN collaborative filtering."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from recomx.models.item_knn import (
    ItemKNNConfig,
    build_item_neighborhoods,
    recommend_item_knn,
)
from recomx.models.sparse_interactions import build_sparse_interaction_matrix


@pytest.fixture
def interactions():
    frame = pd.DataFrame(
        {
            "userId": [1, 1, 2, 2, 3, 3, 4, 4],
            "movieId": [10, 20, 10, 30, 20, 30, 30, 40],
            "rating": [5.0] * 8,
        }
    )
    return build_sparse_interaction_matrix(frame)


def test_item_knn_graph_is_sparse_and_bounded(interactions):
    model = build_item_neighborhoods(interactions, ItemKNNConfig(neighbors=2))
    assert sparse.isspmatrix_csr(model.matrix)
    assert model.matrix.dtype == np.float32
    assert model.matrix.nnz <= interactions.n_items * 2


def test_item_knn_excludes_self_neighbors(interactions):
    model = build_item_neighborhoods(interactions, ItemKNNConfig(neighbors=3))
    assert np.all(model.matrix.diagonal() == 0.0)


def test_neighbor_lookup_uses_external_item_ids(interactions):
    model = build_item_neighborhoods(interactions, ItemKNNConfig(neighbors=2))
    item_ids, scores = model.neighbors_for_item(10)
    assert set(item_ids).issubset({20, 30, 40})
    assert len(item_ids) == len(scores)


def test_unknown_item_returns_empty_neighbors(interactions):
    model = build_item_neighborhoods(interactions)
    item_ids, scores = model.neighbors_for_item(999)
    assert item_ids.size == 0
    assert scores.size == 0


def test_recommendations_filter_seen_items(interactions):
    model = build_item_neighborhoods(interactions, ItemKNNConfig(neighbors=3))
    item_ids, _ = recommend_item_knn(1, interactions, model, top_k=3)
    assert 10 not in item_ids
    assert 20 not in item_ids


def test_unknown_user_returns_empty_for_fallback(interactions):
    model = build_item_neighborhoods(interactions)
    item_ids, scores = recommend_item_knn(999, interactions, model)
    assert item_ids.size == 0
    assert scores.size == 0


def test_recommendations_are_deterministic(interactions):
    model = build_item_neighborhoods(interactions, ItemKNNConfig(neighbors=3))
    first = recommend_item_knn(1, interactions, model, top_k=2)
    second = recommend_item_knn(1, interactions, model, top_k=2)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_invalid_neighbor_count_fails():
    with pytest.raises(ValueError, match="neighbors"):
        ItemKNNConfig(neighbors=0).validate()


def test_invalid_top_k_fails(interactions):
    model = build_item_neighborhoods(interactions)
    with pytest.raises(ValueError, match="top_k"):
        recommend_item_knn(1, interactions, model, top_k=0)
