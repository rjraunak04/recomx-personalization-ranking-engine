"""Tests for collaborative candidate generation and fallback."""

import numpy as np
import pandas as pd

from recomx.models.cf_candidates import (
    generate_item_knn_candidates,
    generate_svd_candidates,
)
from recomx.models.item_knn import ItemKNNConfig, build_item_neighborhoods
from recomx.models.latent_svd import LatentSVDConfig, fit_latent_svd
from recomx.models.popularity import PopularityModel, build_seen_item_map
from recomx.models.sparse_interactions import build_sparse_interaction_matrix


def _fixture():
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
    # Seen-but-nonrelevant item: collaborative history excludes it, final filter must not.
    rows.append((1, 60, 2.0))
    train = pd.DataFrame(rows, columns=["userId", "movieId", "rating"])
    interactions = build_sparse_interaction_matrix(train)
    popularity = PopularityModel.fit(train)
    seen = build_seen_item_map(train)
    return train, interactions, popularity, seen


def test_item_knn_warm_and_cold_strategies():
    _, interactions, popularity, seen = _fixture()
    knn = build_item_neighborhoods(interactions, ItemKNNConfig(neighbors=4))
    recs, summary = generate_item_knn_candidates(
        [1, 999], interactions, knn, popularity, seen, top_k=2
    )
    assert summary.warm_users == 1
    assert summary.cold_users == 1
    assert set(recs["strategy"]) == {"item_knn", "popularity_fallback"}


def test_item_knn_filters_every_train_seen_item():
    _, interactions, popularity, seen = _fixture()
    knn = build_item_neighborhoods(interactions, ItemKNNConfig(neighbors=5))
    recs, summary = generate_item_knn_candidates(
        [1], interactions, knn, popularity, seen, top_k=2
    )
    assert set(recs["movieId"]).isdisjoint(seen[1])
    assert summary.seen_item_violations == 0


def test_svd_warm_and_cold_strategies():
    _, interactions, popularity, seen = _fixture()
    svd = fit_latent_svd(interactions, LatentSVDConfig(factors=2))
    recs, summary = generate_svd_candidates(
        [2, 999], interactions, svd, popularity, seen, top_k=2, batch_size=2
    )
    assert summary.warm_users == 1
    assert summary.cold_users == 1
    assert set(recs["strategy"]) == {"latent_svd", "popularity_fallback"}


def test_svd_filters_every_train_seen_item():
    _, interactions, popularity, seen = _fixture()
    svd = fit_latent_svd(interactions, LatentSVDConfig(factors=2))
    recs, summary = generate_svd_candidates(
        [1], interactions, svd, popularity, seen, top_k=2, batch_size=2
    )
    assert set(recs["movieId"]).isdisjoint(seen[1])
    assert summary.seen_item_violations == 0


def test_candidate_ranks_are_contiguous():
    _, interactions, popularity, seen = _fixture()
    knn = build_item_neighborhoods(interactions, ItemKNNConfig(neighbors=5))
    recs, _ = generate_item_knn_candidates(
        [1, 2], interactions, knn, popularity, seen, top_k=2
    )
    for _, group in recs.groupby("userId"):
        np.testing.assert_array_equal(group["recommendation_rank"], [1, 2])


def test_generation_is_deterministic():
    _, interactions, popularity, seen = _fixture()
    knn = build_item_neighborhoods(interactions, ItemKNNConfig(neighbors=5))
    first, _ = generate_item_knn_candidates(
        [1, 2], interactions, knn, popularity, seen, top_k=2
    )
    second, _ = generate_item_knn_candidates(
        [1, 2], interactions, knn, popularity, seen, top_k=2
    )
    pd.testing.assert_frame_equal(first, second)
