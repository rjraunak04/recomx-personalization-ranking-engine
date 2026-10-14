"""Tests for the Day-03 sparse interaction representation."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from recomx.models.sparse_interactions import build_sparse_interaction_matrix


@pytest.fixture
def train() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "userId": [20, 10, 20, 10, 30, 30, 20],
            "movieId": [300, 100, 100, 300, 200, 400, 300],
            "rating": [5.0, 4.5, 3.0, 4.0, 2.0, 5.0, 4.5],
        }
    )


def test_builds_csr_float32_binary_matrix(train: pd.DataFrame) -> None:
    result = build_sparse_interaction_matrix(train)

    assert sparse.isspmatrix_csr(result.matrix)
    assert result.matrix.dtype == np.float32
    assert np.all(result.matrix.data == 1.0)


def test_only_relevant_feedback_is_encoded(train: pd.DataFrame) -> None:
    result = build_sparse_interaction_matrix(train)

    assert result.n_interactions == 4
    assert set(result.positive_items_for_user(20)) == {300}
    assert set(result.positive_items_for_user(30)) == {400}


def test_duplicate_positive_pairs_are_collapsed(train: pd.DataFrame) -> None:
    result = build_sparse_interaction_matrix(train)
    assert result.n_interactions == 4


def test_mappings_are_deterministic(train: pd.DataFrame) -> None:
    first = build_sparse_interaction_matrix(train)
    shuffled = train.sample(frac=1.0, random_state=7)
    second = build_sparse_interaction_matrix(shuffled)

    assert first.user_ids.tolist() == [10, 20, 30]
    assert first.item_ids.tolist() == [100, 300, 400]
    assert first.user_to_index == second.user_to_index
    assert first.item_to_index == second.item_to_index
    assert (first.matrix != second.matrix).nnz == 0


def test_unknown_user_has_empty_history(train: pd.DataFrame) -> None:
    result = build_sparse_interaction_matrix(train)
    values = result.positive_items_for_user(999)

    assert values.dtype == np.int64
    assert values.size == 0


def test_density_uses_observed_positive_pairs(train: pd.DataFrame) -> None:
    result = build_sparse_interaction_matrix(train)
    assert result.density == pytest.approx(4 / 9)


def test_missing_columns_fail() -> None:
    frame = pd.DataFrame({"userId": [1], "movieId": [2]})

    with pytest.raises(ValueError, match="missing required columns"):
        build_sparse_interaction_matrix(frame)


def test_empty_training_data_fails() -> None:
    frame = pd.DataFrame(columns=["userId", "movieId", "rating"])

    with pytest.raises(ValueError, match="cannot be empty"):
        build_sparse_interaction_matrix(frame)


def test_no_relevant_interactions_fail() -> None:
    frame = pd.DataFrame({"userId": [1], "movieId": [2], "rating": [2.0]})

    with pytest.raises(ValueError, match="No relevant"):
        build_sparse_interaction_matrix(frame)
