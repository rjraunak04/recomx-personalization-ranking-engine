"""Day-06 serving contract for personalized recommendations."""
from __future__ import annotations

from dataclasses import dataclass, field

MAX_TOP_K = 20

@dataclass(frozen=True)
class RecommendationRequest:
    user_id: int
    top_k: int = MAX_TOP_K

    def validate(self) -> None:
        if self.user_id <= 0:
            raise ValueError("user_id must be a positive integer.")
        if not 1 <= self.top_k <= MAX_TOP_K:
            raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}.")

@dataclass(frozen=True)
class RecommendationItem:
    movie_id: int
    rank: int
    score: float
    reason: str

@dataclass(frozen=True)
class RecommendationResponse:
    user_id: int
    strategy: str
    recommendations: tuple[RecommendationItem, ...] = field(default_factory=tuple)

    def validate(self) -> None:
        if self.user_id <= 0:
            raise ValueError("user_id must be a positive integer.")
        if self.strategy not in {"personalized", "cold_start_popularity"}:
            raise ValueError("Unsupported recommendation strategy.")
        ranks = [item.rank for item in self.recommendations]
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError("Recommendation ranks must be contiguous from 1.")
        movie_ids = [item.movie_id for item in self.recommendations]
        if len(movie_ids) != len(set(movie_ids)):
            raise ValueError("Recommendation movie IDs must be unique.")
