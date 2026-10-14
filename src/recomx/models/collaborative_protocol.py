"""Immutable Day-03 collaborative-filtering experiment contract."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CollaborativeFilteringProtocol:
    """Rules shared by Day-03 collaborative recommenders."""

    task: str = "personalized_top_k"
    training_partition: str = "train"
    selection_partition: str = "validation"
    locked_test_access: bool = False

    feedback_type: str = "implicit_positive"
    relevance_threshold: float = 4.0
    positive_value: float = 1.0

    matrix_format: str = "csr"
    matrix_dtype: str = "float32"
    deterministic_mappings: bool = True
    allow_dense_matrix: bool = False

    filter_seen_items: bool = True
    cold_user_fallback: str = "popularity"
    top_k: tuple[int, ...] = (5, 10, 20)

    item_knn_metric: str = "cosine"
    item_knn_neighbors: int = 100

    latent_algorithm: str = "truncated_svd"
    latent_factors: int = 64
    random_state: int = 42

    def validate(self) -> None:
        """Fail fast when a setting breaks the Day-03 experiment contract."""

        if self.task != "personalized_top_k":
            raise ValueError("Day-03 task must remain personalized_top_k.")
        if self.training_partition != "train":
            raise ValueError("Collaborative models must fit on train only.")
        if self.selection_partition != "validation":
            raise ValueError("Model selection must use validation only.")
        if self.locked_test_access:
            raise ValueError("The locked test partition must remain inaccessible on Day-03.")

        if self.feedback_type != "implicit_positive":
            raise ValueError("Day-03 retrieval uses binary positive implicit feedback.")
        if not 0.5 <= self.relevance_threshold <= 5.0:
            raise ValueError("Relevance threshold must lie on the MovieLens rating scale.")
        if self.positive_value != 1.0:
            raise ValueError("Positive sparse interactions must have value 1.0.")

        if self.matrix_format != "csr":
            raise ValueError("The canonical Day-03 user-item matrix must use CSR format.")
        if self.matrix_dtype != "float32":
            raise ValueError("The canonical sparse matrix must use float32.")
        if not self.deterministic_mappings:
            raise ValueError("User and item mappings must be deterministic.")
        if self.allow_dense_matrix:
            raise ValueError("Dense user-item matrices are forbidden at MovieLens 25M scale.")

        if not self.filter_seen_items:
            raise ValueError("All train-seen items must be filtered before recommendation.")
        if self.cold_user_fallback != "popularity":
            raise ValueError("Cold users must use the frozen train-only popularity fallback.")

        if not self.top_k or any(k <= 0 for k in self.top_k):
            raise ValueError("Top-K cutoffs must be positive.")
        if tuple(sorted(set(self.top_k))) != self.top_k:
            raise ValueError("Top-K cutoffs must be unique and strictly increasing.")

        if self.item_knn_metric != "cosine":
            raise ValueError("The Day-03 Item-KNN contract uses cosine similarity.")
        if self.item_knn_neighbors <= 0:
            raise ValueError("Item-KNN neighbors must be positive.")

        if self.latent_algorithm != "truncated_svd":
            raise ValueError("The Day-03 latent baseline must use truncated_svd.")
        if self.latent_factors <= 0:
            raise ValueError("Latent factor count must be positive.")
        if self.random_state < 0:
            raise ValueError("random_state must be non-negative.")


DEFAULT_COLLABORATIVE_PROTOCOL = CollaborativeFilteringProtocol()
DEFAULT_COLLABORATIVE_PROTOCOL.validate()
