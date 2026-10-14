"""Tests for the Day-03 latent SVD retrieval model."""

import numpy as np
import pandas as pd
import pytest

from recomx.models.latent_svd import LatentSVDConfig, fit_latent_svd
from recomx.models.sparse_interactions import build_sparse_interaction_matrix


@pytest.fixture
def interactions():
    rows = []
    positives = {
        1: [10, 20, 30],
        2: [10, 20, 40],
        3: [20, 30, 50],
        4: [30, 40, 50],
        5: [10, 40, 60],
        6: [20, 50, 60],
    }
    for user_id, items in positives.items():
        rows.extend((user_id, item_id, 5.0) for item_id in items)
    return build_sparse_interaction_matrix(
        pd.DataFrame(rows, columns=["userId", "movieId", "rating"])
    )


def test_fit_shapes_and_float32(interactions):
    model = fit_latent_svd(interactions, LatentSVDConfig(factors=2, random_state=42))
    assert model.user_factors.shape == (6, 2)
    assert model.item_factors.shape == (6, 2)
    assert model.user_factors.dtype == np.float32
    assert model.item_factors.dtype == np.float32


def test_factor_count_is_exposed(interactions):
    model = fit_latent_svd(interactions, LatentSVDConfig(factors=2))
    assert model.n_factors == 2


def test_recommendations_filter_seen_items(interactions):
    model = fit_latent_svd(interactions, LatentSVDConfig(factors=2))
    item_ids, scores = model.recommend(1, interactions, top_k=3, batch_size=2)
    assert set(item_ids).isdisjoint({10, 20, 30})
    assert len(item_ids) == len(scores)


def test_unknown_user_returns_empty_for_fallback(interactions):
    model = fit_latent_svd(interactions, LatentSVDConfig(factors=2))
    item_ids, scores = model.recommend(999, interactions)
    assert item_ids.size == 0
    assert scores.size == 0


def test_fit_is_deterministic(interactions):
    config = LatentSVDConfig(factors=2, random_state=42)
    first = fit_latent_svd(interactions, config)
    second = fit_latent_svd(interactions, config)
    np.testing.assert_allclose(first.user_factors, second.user_factors, atol=1e-5)
    np.testing.assert_allclose(first.item_factors, second.item_factors, atol=1e-5)


def test_invalid_factor_count_fails(interactions):
    with pytest.raises(ValueError, match="smaller"):
        fit_latent_svd(interactions, LatentSVDConfig(factors=6))


def test_invalid_top_k_fails(interactions):
    model = fit_latent_svd(interactions, LatentSVDConfig(factors=2))
    with pytest.raises(ValueError, match="top_k"):
        model.recommend(1, interactions, top_k=0)


def test_invalid_batch_size_fails(interactions):
    model = fit_latent_svd(interactions, LatentSVDConfig(factors=2))
    with pytest.raises(ValueError, match="batch_size"):
        model.recommend(1, interactions, batch_size=0)
