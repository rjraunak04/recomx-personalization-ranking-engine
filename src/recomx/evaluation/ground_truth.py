"""Leakage-safe validation ground-truth construction for RecomX."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from recomx.evaluation.relevance import RelevancePolicy

_REQUIRED_COLUMNS = {"userId", "movieId", "rating"}


@dataclass(frozen=True)
class GroundTruthSummary:
    """Summary of validation ground truth and evaluation cohorts."""

    train_users: int
    validation_users: int
    validation_relevant_users: int
    warm_validation_users: int
    cold_validation_users: int
    evaluable_warm_users: int
    evaluable_cold_users: int
    users_without_relevant_targets: int
    relevant_validation_interactions: int
    seen_relevant_interactions_removed: int
    ground_truth_interactions: int
    ground_truth_items: int

    def as_dict(self) -> dict[str, int]:
        """Return serialization-friendly summary."""

        return {
            "train_users": self.train_users,
            "validation_users": self.validation_users,
            "validation_relevant_users": self.validation_relevant_users,
            "warm_validation_users": self.warm_validation_users,
            "cold_validation_users": self.cold_validation_users,
            "evaluable_warm_users": self.evaluable_warm_users,
            "evaluable_cold_users": self.evaluable_cold_users,
            "users_without_relevant_targets": (self.users_without_relevant_targets),
            "relevant_validation_interactions": (self.relevant_validation_interactions),
            "seen_relevant_interactions_removed": (self.seen_relevant_interactions_removed),
            "ground_truth_interactions": self.ground_truth_interactions,
            "ground_truth_items": self.ground_truth_items,
        }


@dataclass(frozen=True)
class GroundTruthResult:
    """Validation ground truth together with user cohorts."""

    ground_truth: pd.DataFrame
    cohorts: pd.DataFrame
    summary: GroundTruthSummary


def _validate_frame(
    frame: pd.DataFrame,
    name: str,
) -> None:
    """Validate required interaction columns."""

    missing = _REQUIRED_COLUMNS.difference(frame.columns)

    if missing:
        raise ValueError(f"{name} missing required columns: {sorted(missing)}")

    if frame.empty:
        raise ValueError(f"{name} cannot be empty.")

    if frame[["userId", "movieId"]].isna().any().any():
        raise ValueError(f"{name} contains missing userId or movieId values.")


def build_validation_ground_truth(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    policy: RelevancePolicy | None = None,
) -> GroundTruthResult:
    """Build leakage-safe validation targets and evaluation cohorts.

    Relevant validation interactions are defined by the explicit-feedback
    relevance policy. For warm users, relevant items already observed in
    training are removed from the evaluation target.

    Cold users have no training history, so they have no seen training items
    to remove.
    """

    _validate_frame(train, "train")
    _validate_frame(validation, "validation")

    active_policy = policy or RelevancePolicy()
    active_policy.validate()

    relevant_validation = active_policy.relevant_interactions(validation)

    train_users = pd.Index(train["userId"].unique())

    validation_users = pd.Index(validation["userId"].unique())

    warm_mask = validation["userId"].isin(train_users)

    warm_users = pd.Index(validation.loc[warm_mask, "userId"].unique())

    cold_users = pd.Index(validation.loc[~warm_mask, "userId"].unique())

    seen_pairs = train[["userId", "movieId"]].drop_duplicates()

    relevant_with_seen = relevant_validation.merge(
        seen_pairs.assign(_seen_in_train=True),
        on=["userId", "movieId"],
        how="left",
        validate="many_to_one",
    )

    seen_relevant_mask = relevant_with_seen["_seen_in_train"].fillna(False).astype(bool)

    ground_truth = (
        relevant_with_seen.loc[
            ~seen_relevant_mask,
            ["userId", "movieId", "rating"],
        ]
        .drop_duplicates(subset=["userId", "movieId"])
        .sort_values(
            ["userId", "movieId"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    relevant_users = pd.Index(relevant_validation["userId"].unique())

    evaluable_users = pd.Index(ground_truth["userId"].unique())

    cohort_frame = pd.DataFrame(
        {
            "userId": validation_users,
        }
    )

    cohort_frame["is_warm_user"] = cohort_frame["userId"].isin(warm_users)

    cohort_frame["is_cold_user"] = cohort_frame["userId"].isin(cold_users)

    cohort_frame["has_relevant_validation_target"] = cohort_frame["userId"].isin(relevant_users)

    cohort_frame["is_evaluable"] = cohort_frame["userId"].isin(evaluable_users)

    cohort_frame = cohort_frame.sort_values(
        "userId",
        kind="mergesort",
    ).reset_index(drop=True)

    evaluable_warm_users = int((cohort_frame["is_warm_user"] & cohort_frame["is_evaluable"]).sum())

    evaluable_cold_users = int((cohort_frame["is_cold_user"] & cohort_frame["is_evaluable"]).sum())

    summary = GroundTruthSummary(
        train_users=int(len(train_users)),
        validation_users=int(len(validation_users)),
        validation_relevant_users=int(len(relevant_users)),
        warm_validation_users=int(len(warm_users)),
        cold_validation_users=int(len(cold_users)),
        evaluable_warm_users=evaluable_warm_users,
        evaluable_cold_users=evaluable_cold_users,
        users_without_relevant_targets=int((~cohort_frame["has_relevant_validation_target"]).sum()),
        relevant_validation_interactions=int(len(relevant_validation)),
        seen_relevant_interactions_removed=int(seen_relevant_mask.sum()),
        ground_truth_interactions=int(len(ground_truth)),
        ground_truth_items=int(ground_truth["movieId"].nunique()),
    )

    return GroundTruthResult(
        ground_truth=ground_truth,
        cohorts=cohort_frame,
        summary=summary,
    )
