"""Tests for the Day-03 collaborative-filtering contract."""

from dataclasses import replace

import pytest

from recomx.models.collaborative_protocol import (
    DEFAULT_COLLABORATIVE_PROTOCOL,
)


def test_default_collaborative_protocol_is_valid() -> None:
    DEFAULT_COLLABORATIVE_PROTOCOL.validate()


def test_day02_evaluation_semantics_are_preserved() -> None:
    protocol = DEFAULT_COLLABORATIVE_PROTOCOL
    assert protocol.training_partition == "train"
    assert protocol.selection_partition == "validation"
    assert protocol.relevance_threshold == 4.0
    assert protocol.top_k == (5, 10, 20)


def test_locked_test_access_is_forbidden() -> None:
    protocol = replace(DEFAULT_COLLABORATIVE_PROTOCOL, locked_test_access=True)
    with pytest.raises(ValueError, match="locked test"):
        protocol.validate()


def test_dense_matrix_is_forbidden() -> None:
    protocol = replace(DEFAULT_COLLABORATIVE_PROTOCOL, allow_dense_matrix=True)
    with pytest.raises(ValueError, match="Dense user-item"):
        protocol.validate()


def test_sparse_representation_is_fixed() -> None:
    protocol = DEFAULT_COLLABORATIVE_PROTOCOL
    assert protocol.matrix_format == "csr"
    assert protocol.matrix_dtype == "float32"
    assert protocol.deterministic_mappings is True


def test_feedback_policy_is_binary_positive() -> None:
    protocol = DEFAULT_COLLABORATIVE_PROTOCOL
    assert protocol.feedback_type == "implicit_positive"
    assert protocol.positive_value == 1.0


def test_seen_item_filtering_is_required() -> None:
    protocol = replace(DEFAULT_COLLABORATIVE_PROTOCOL, filter_seen_items=False)
    with pytest.raises(ValueError, match="train-seen"):
        protocol.validate()


def test_cold_users_use_popularity_fallback() -> None:
    protocol = replace(DEFAULT_COLLABORATIVE_PROTOCOL, cold_user_fallback="none")
    with pytest.raises(ValueError, match="Cold users"):
        protocol.validate()


def test_item_knn_configuration_is_validated() -> None:
    protocol = replace(DEFAULT_COLLABORATIVE_PROTOCOL, item_knn_neighbors=0)
    with pytest.raises(ValueError, match="neighbors"):
        protocol.validate()


def test_latent_factor_configuration_is_validated() -> None:
    protocol = replace(DEFAULT_COLLABORATIVE_PROTOCOL, latent_factors=0)
    with pytest.raises(ValueError, match="factor"):
        protocol.validate()


def test_top_k_must_be_strictly_increasing() -> None:
    protocol = replace(DEFAULT_COLLABORATIVE_PROTOCOL, top_k=(10, 5, 20))
    with pytest.raises(ValueError, match="strictly increasing"):
        protocol.validate()
