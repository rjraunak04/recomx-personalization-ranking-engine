"""Tests for the RecomX recommendation evaluation contract."""

from dataclasses import replace

import pytest

from recomx.evaluation.protocol import (
    DEFAULT_EVALUATION_PROTOCOL,
    EvaluationProtocol,
)


def test_default_protocol_is_valid() -> None:
    DEFAULT_EVALUATION_PROTOCOL.validate()


def test_default_partitions_are_leakage_safe() -> None:
    protocol = DEFAULT_EVALUATION_PROTOCOL

    assert protocol.training_partition == "train"
    assert protocol.selection_partition == "validation"
    assert protocol.final_partition == "locked_test"


def test_seen_items_are_filtered() -> None:
    assert DEFAULT_EVALUATION_PROTOCOL.filter_seen_items is True


def test_future_information_is_forbidden() -> None:
    assert DEFAULT_EVALUATION_PROTOCOL.allow_future_information is False


def test_default_relevance_threshold() -> None:
    assert DEFAULT_EVALUATION_PROTOCOL.relevance_threshold == 4.0


def test_default_top_k_cutoffs() -> None:
    assert DEFAULT_EVALUATION_PROTOCOL.top_k == (5, 10, 20)


def test_future_information_fails_validation() -> None:
    protocol = replace(
        DEFAULT_EVALUATION_PROTOCOL,
        allow_future_information=True,
    )

    with pytest.raises(ValueError, match="Future information"):
        protocol.validate()


def test_seen_item_filter_cannot_be_disabled() -> None:
    protocol = replace(
        DEFAULT_EVALUATION_PROTOCOL,
        filter_seen_items=False,
    )

    with pytest.raises(ValueError, match="Seen training items"):
        protocol.validate()


def test_locked_test_cannot_be_replaced() -> None:
    protocol = replace(
        DEFAULT_EVALUATION_PROTOCOL,
        final_partition="validation",
    )

    with pytest.raises(ValueError, match="locked_test"):
        protocol.validate()


def test_invalid_top_k_fails() -> None:
    protocol = EvaluationProtocol(top_k=(5, 0, 20))

    with pytest.raises(ValueError, match="positive"):
        protocol.validate()


def test_unsorted_top_k_fails() -> None:
    protocol = EvaluationProtocol(top_k=(10, 5, 20))

    with pytest.raises(ValueError, match="strictly increasing"):
        protocol.validate()
