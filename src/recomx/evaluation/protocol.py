"""Evaluation protocol for RecomX recommendation experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationProtocol:
    """Immutable rules governing offline recommendation evaluation."""

    task: str = "personalized_top_k"
    training_partition: str = "train"
    selection_partition: str = "validation"
    final_partition: str = "locked_test"

    filter_seen_items: bool = True
    chronological_evaluation: bool = True
    allow_future_information: bool = False

    relevance_threshold: float = 4.0
    top_k: tuple[int, ...] = (5, 10, 20)

    def validate(self) -> None:
        """Validate the recommendation evaluation contract."""

        if self.task != "personalized_top_k":
            raise ValueError("Day-02 task must be personalized_top_k.")

        if self.training_partition != "train":
            raise ValueError("Models must train only on the train partition.")

        if self.selection_partition != "validation":
            raise ValueError("Model-selection decisions must use validation only.")

        if self.final_partition != "locked_test":
            raise ValueError("Final evaluation partition must be locked_test.")

        if not self.filter_seen_items:
            raise ValueError("Seen training items must be filtered.")

        if not self.chronological_evaluation:
            raise ValueError("Evaluation must remain chronological.")

        if self.allow_future_information:
            raise ValueError("Future information is forbidden.")

        if not 0.5 <= self.relevance_threshold <= 5.0:
            raise ValueError("Relevance threshold must lie on the MovieLens rating scale.")

        if not self.top_k:
            raise ValueError("At least one Top-K cutoff is required.")

        if any(k <= 0 for k in self.top_k):
            raise ValueError("Top-K values must be positive.")

        if tuple(sorted(set(self.top_k))) != self.top_k:
            raise ValueError("Top-K values must be unique and strictly increasing.")


DEFAULT_EVALUATION_PROTOCOL = EvaluationProtocol()
DEFAULT_EVALUATION_PROTOCOL.validate()
