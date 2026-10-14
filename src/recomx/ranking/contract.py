"""Leakage-safe Day-05 ranking contract."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RankingContract:
    relevance_threshold: float = 4.0
    candidate_budget: int = 100
    final_top_k: int = 20
    locked_test_access: bool = False

    def validate(self) -> None:
        if not 0.5 <= self.relevance_threshold <= 5.0:
            raise ValueError("relevance_threshold must lie in [0.5, 5.0].")
        if self.candidate_budget <= 0:
            raise ValueError("candidate_budget must be positive.")
        if self.final_top_k <= 0:
            raise ValueError("final_top_k must be positive.")
        if self.final_top_k > self.candidate_budget:
            raise ValueError("final_top_k cannot exceed candidate_budget.")
        if self.locked_test_access:
            raise ValueError("Locked test access is forbidden during Day-05 development.")
