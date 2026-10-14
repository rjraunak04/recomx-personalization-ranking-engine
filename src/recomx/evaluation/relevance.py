"""Explicit-feedback relevance policy for RecomX."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RelevancePolicy:
    """Convert explicit MovieLens ratings into relevance labels."""

    minimum_rating: float = 4.0
    rating_column: str = "rating"

    def validate(self) -> None:
        """Validate policy configuration."""

        if not 0.5 <= self.minimum_rating <= 5.0:
            raise ValueError("minimum_rating must lie on the MovieLens rating scale.")

        if not self.rating_column:
            raise ValueError("rating_column cannot be empty.")

    def mask(self, frame: pd.DataFrame) -> pd.Series:
        """Return a Boolean mask identifying relevant interactions."""

        self.validate()

        if self.rating_column not in frame.columns:
            raise ValueError(f"Missing required rating column: {self.rating_column}")

        ratings = frame[self.rating_column]

        if ratings.isna().any():
            raise ValueError("Ratings contain missing values.")

        return ratings.ge(self.minimum_rating)

    def relevant_interactions(
        self,
        frame: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return relevant interactions without mutating the input."""

        return frame.loc[self.mask(frame)].copy()


@dataclass(frozen=True)
class RelevanceProfile:
    """Summary of relevance-policy behaviour on interaction data."""

    total_interactions: int
    relevant_interactions: int
    non_relevant_interactions: int
    relevance_rate: float
    total_users: int
    relevant_users: int
    total_items: int
    relevant_items: int
    minimum_rating: float

    def as_dict(self) -> dict[str, int | float]:
        """Return a serialization-friendly representation."""

        return {
            "total_interactions": self.total_interactions,
            "relevant_interactions": self.relevant_interactions,
            "non_relevant_interactions": self.non_relevant_interactions,
            "relevance_rate": self.relevance_rate,
            "total_users": self.total_users,
            "relevant_users": self.relevant_users,
            "total_items": self.total_items,
            "relevant_items": self.relevant_items,
            "minimum_rating": self.minimum_rating,
        }


def build_relevance_profile(
    frame: pd.DataFrame,
    policy: RelevancePolicy | None = None,
) -> RelevanceProfile:
    """Profile explicit interactions under a relevance policy."""

    active_policy = policy or RelevancePolicy()
    active_policy.validate()

    required = {
        "userId",
        "movieId",
        active_policy.rating_column,
    }

    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if frame.empty:
        raise ValueError("Cannot profile an empty interaction table.")

    relevant_mask = active_policy.mask(frame)
    relevant = frame.loc[relevant_mask]

    total_interactions = len(frame)
    relevant_interactions = int(relevant_mask.sum())

    return RelevanceProfile(
        total_interactions=total_interactions,
        relevant_interactions=relevant_interactions,
        non_relevant_interactions=(total_interactions - relevant_interactions),
        relevance_rate=(relevant_interactions / total_interactions),
        total_users=int(frame["userId"].nunique()),
        relevant_users=int(relevant["userId"].nunique()),
        total_items=int(frame["movieId"].nunique()),
        relevant_items=int(relevant["movieId"].nunique()),
        minimum_rating=active_policy.minimum_rating,
    )
