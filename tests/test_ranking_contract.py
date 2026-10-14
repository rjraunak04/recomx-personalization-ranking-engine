from __future__ import annotations

import pytest

from recomx.ranking.contract import RankingContract


def test_default_contract_is_valid() -> None:
    RankingContract().validate()


def test_final_k_cannot_exceed_candidate_budget() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        RankingContract(candidate_budget=10, final_top_k=20).validate()


def test_locked_test_access_is_rejected() -> None:
    with pytest.raises(ValueError, match="Locked test"):
        RankingContract(locked_test_access=True).validate()


@pytest.mark.parametrize("threshold", [0.0, 5.5])
def test_invalid_relevance_threshold_is_rejected(threshold: float) -> None:
    with pytest.raises(ValueError, match="relevance_threshold"):
        RankingContract(relevance_threshold=threshold).validate()
