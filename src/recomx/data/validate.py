"""Recommendation-specific data quality validation for MovieLens 25M."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ValidationResult:
    """Result of one data-quality validation rule."""

    rule: str
    passed: bool
    value: Any
    message: str


class DataQualityError(RuntimeError):
    """Raised when a required RecomX data-quality rule fails."""


def _result(
    rule: str,
    passed: bool,
    value: Any,
    message: str,
) -> ValidationResult:
    """Create a standardized validation result."""

    return ValidationResult(
        rule=rule,
        passed=passed,
        value=value,
        message=message,
    )


def validate_ratings(
    frame: pd.DataFrame,
) -> list[ValidationResult]:
    """Validate the primary user-item rating interaction table."""

    results: list[ValidationResult] = []

    required = {
        "userId",
        "movieId",
        "rating",
        "timestamp",
    }

    missing = sorted(required - set(frame.columns))

    results.append(
        _result(
            "ratings_required_columns",
            not missing,
            missing,
            "Required ratings columns must be present.",
        )
    )

    if missing:
        return results

    nulls = int(frame[list(required)].isna().sum().sum())

    results.append(
        _result(
            "ratings_required_nulls",
            nulls == 0,
            nulls,
            "Required rating fields must not contain null values.",
        )
    )

    invalid_users = int((frame["userId"] <= 0).sum())

    results.append(
        _result(
            "ratings_user_ids",
            invalid_users == 0,
            invalid_users,
            "User identifiers must be positive.",
        )
    )

    invalid_movies = int((frame["movieId"] <= 0).sum())

    results.append(
        _result(
            "ratings_movie_ids",
            invalid_movies == 0,
            invalid_movies,
            "Movie identifiers must be positive.",
        )
    )

    invalid_ratings = int((~frame["rating"].between(0.5, 5.0)).sum())

    results.append(
        _result(
            "ratings_domain",
            invalid_ratings == 0,
            invalid_ratings,
            "Ratings must remain within the MovieLens 0.5-5.0 domain.",
        )
    )

    doubled_rating = frame["rating"] * 2

    invalid_half_steps = int((doubled_rating.round() - doubled_rating).abs().gt(1e-6).sum())

    results.append(
        _result(
            "ratings_half_star_scale",
            invalid_half_steps == 0,
            invalid_half_steps,
            "Ratings must use valid half-star increments.",
        )
    )

    duplicate_events = int(frame.duplicated().sum())

    results.append(
        _result(
            "ratings_exact_duplicates",
            duplicate_events == 0,
            duplicate_events,
            "Exact duplicate rating events must not exist.",
        )
    )

    repeated_pairs = int(
        frame.duplicated(
            subset=["userId", "movieId"],
            keep=False,
        ).sum()
    )

    results.append(
        _result(
            "ratings_repeated_user_movie_rows",
            True,
            repeated_pairs,
            (
                "Repeated user-movie rows are measured rather than "
                "automatically rejected because temporal interaction "
                "handling is a downstream modeling decision."
            ),
        )
    )

    invalid_timestamps = int(frame["timestamp"].isna().sum())

    results.append(
        _result(
            "ratings_timestamps",
            invalid_timestamps == 0,
            invalid_timestamps,
            ("Rating timestamps must be available for temporal evaluation."),
        )
    )

    return results


def validate_movies(
    frame: pd.DataFrame,
) -> list[ValidationResult]:
    """Validate the movie catalogue."""

    results: list[ValidationResult] = []

    required = {
        "movieId",
        "title",
        "genres",
    }

    missing = sorted(required - set(frame.columns))

    results.append(
        _result(
            "movies_required_columns",
            not missing,
            missing,
            "Required movie catalogue columns must be present.",
        )
    )

    if missing:
        return results

    null_ids = int(frame["movieId"].isna().sum())

    results.append(
        _result(
            "movies_id_nulls",
            null_ids == 0,
            null_ids,
            "Movie identifiers must not be null.",
        )
    )

    invalid_ids = int((frame["movieId"] <= 0).sum())

    results.append(
        _result(
            "movies_positive_ids",
            invalid_ids == 0,
            invalid_ids,
            "Movie identifiers must be positive.",
        )
    )

    duplicate_ids = int(frame["movieId"].duplicated().sum())

    results.append(
        _result(
            "movies_unique_ids",
            duplicate_ids == 0,
            duplicate_ids,
            "Each catalogue movie must have a unique movieId.",
        )
    )

    blank_titles = int(frame["title"].fillna("").str.strip().eq("").sum())

    results.append(
        _result(
            "movies_titles",
            blank_titles == 0,
            blank_titles,
            "Movie titles must not be blank.",
        )
    )

    missing_genres = int(frame["genres"].isna().sum())

    results.append(
        _result(
            "movies_genres_not_null",
            missing_genres == 0,
            missing_genres,
            "Genre field must be structurally available.",
        )
    )

    return results


def validate_tags(
    frame: pd.DataFrame,
) -> list[ValidationResult]:
    """Validate user-generated tag interactions."""

    results: list[ValidationResult] = []

    required = {
        "userId",
        "movieId",
        "tag",
        "timestamp",
    }

    missing = sorted(required - set(frame.columns))

    results.append(
        _result(
            "tags_required_columns",
            not missing,
            missing,
            "Required tag interaction columns must be present.",
        )
    )

    if missing:
        return results

    required_nulls = int(
        frame[
            [
                "userId",
                "movieId",
                "timestamp",
            ]
        ]
        .isna()
        .sum()
        .sum()
    )

    results.append(
        _result(
            "tags_required_nulls",
            required_nulls == 0,
            required_nulls,
            ("Tag interaction identifiers and timestamps must be available."),
        )
    )

    invalid_users = int((frame["userId"] <= 0).sum())

    results.append(
        _result(
            "tags_user_ids",
            invalid_users == 0,
            invalid_users,
            "Tag user identifiers must be positive.",
        )
    )

    invalid_movies = int((frame["movieId"] <= 0).sum())

    results.append(
        _result(
            "tags_movie_ids",
            invalid_movies == 0,
            invalid_movies,
            "Tag movie identifiers must be positive.",
        )
    )

    blank_tags = int(frame["tag"].fillna("").str.strip().eq("").sum())

    # Blank tags are an observed auxiliary-content quality issue.
    # They do not invalidate the primary ratings interaction dataset.
    # They will be excluded when model-ready tag features are built.
    results.append(
        _result(
            "tags_nonempty_observation",
            True,
            blank_tags,
            (
                "Blank tags are measured as a non-blocking auxiliary "
                "content-quality issue and will be excluded from "
                "model-ready tag features."
            ),
        )
    )

    invalid_timestamps = int(frame["timestamp"].isna().sum())

    results.append(
        _result(
            "tags_timestamps",
            invalid_timestamps == 0,
            invalid_timestamps,
            "Tag timestamps must be available.",
        )
    )

    return results


def validate_genome(
    scores: pd.DataFrame,
    tags: pd.DataFrame,
) -> list[ValidationResult]:
    """Validate genome relevance features and vocabulary."""

    results: list[ValidationResult] = []

    score_required = {
        "movieId",
        "tagId",
        "relevance",
    }

    tag_required = {
        "tagId",
        "tag",
    }

    missing_scores = sorted(score_required - set(scores.columns))

    missing_tags = sorted(tag_required - set(tags.columns))

    results.append(
        _result(
            "genome_score_required_columns",
            not missing_scores,
            missing_scores,
            "Required genome score columns must be present.",
        )
    )

    results.append(
        _result(
            "genome_tag_required_columns",
            not missing_tags,
            missing_tags,
            "Required genome tag columns must be present.",
        )
    )

    if missing_scores or missing_tags:
        return results

    score_nulls = int(
        scores[
            [
                "movieId",
                "tagId",
                "relevance",
            ]
        ]
        .isna()
        .sum()
        .sum()
    )

    results.append(
        _result(
            "genome_score_nulls",
            score_nulls == 0,
            score_nulls,
            "Genome score fields must not contain null values.",
        )
    )

    invalid_relevance = int(
        (
            ~scores["relevance"].between(
                0.0,
                1.0,
            )
        ).sum()
    )

    results.append(
        _result(
            "genome_relevance_domain",
            invalid_relevance == 0,
            invalid_relevance,
            "Genome relevance values must be between 0 and 1.",
        )
    )

    duplicate_score_keys = int(
        scores.duplicated(
            subset=[
                "movieId",
                "tagId",
            ]
        ).sum()
    )

    results.append(
        _result(
            "genome_score_keys",
            duplicate_score_keys == 0,
            duplicate_score_keys,
            "Each movie-tag genome pair must be unique.",
        )
    )

    duplicate_tag_ids = int(tags["tagId"].duplicated().sum())

    results.append(
        _result(
            "genome_tag_ids",
            duplicate_tag_ids == 0,
            duplicate_tag_ids,
            "Genome tag IDs must be unique.",
        )
    )

    blank_genome_tags = int(tags["tag"].fillna("").str.strip().eq("").sum())

    results.append(
        _result(
            "genome_tag_text",
            blank_genome_tags == 0,
            blank_genome_tags,
            "Genome tag vocabulary text must not be blank.",
        )
    )

    known_tags = set(tags["tagId"])

    orphan_tags = int((~scores["tagId"].isin(known_tags)).sum())

    results.append(
        _result(
            "genome_tag_fk",
            orphan_tags == 0,
            orphan_tags,
            ("Genome score tag IDs must exist in the genome tag vocabulary."),
        )
    )

    return results


def validate_relationships(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    tags: pd.DataFrame,
    genome_scores: pd.DataFrame,
) -> list[ValidationResult]:
    """Validate relationships between recommendation tables."""

    results: list[ValidationResult] = []

    movie_ids = set(movies["movieId"].dropna())

    orphan_ratings = int((~ratings["movieId"].isin(movie_ids)).sum())

    results.append(
        _result(
            "ratings_movie_fk",
            orphan_ratings == 0,
            orphan_ratings,
            ("Rated movies must exist in the movie catalogue."),
        )
    )

    orphan_tags = int((~tags["movieId"].isin(movie_ids)).sum())

    results.append(
        _result(
            "tags_movie_fk",
            orphan_tags == 0,
            orphan_tags,
            ("Tagged movies must exist in the movie catalogue."),
        )
    )

    orphan_genome = int((~genome_scores["movieId"].isin(movie_ids)).sum())

    results.append(
        _result(
            "genome_movie_fk",
            orphan_genome == 0,
            orphan_genome,
            ("Genome movies must exist in the movie catalogue."),
        )
    )

    rated_catalogue_movies = int(ratings["movieId"].nunique())

    catalogue_movies = int(movies["movieId"].nunique())

    coverage = rated_catalogue_movies / catalogue_movies if catalogue_movies else 0.0

    results.append(
        _result(
            "catalogue_rating_coverage",
            0.0 <= coverage <= 1.0,
            round(coverage, 6),
            ("Fraction of catalogue movies represented in the ratings interaction table."),
        )
    )

    return results


def assert_required_rules(
    results: list[ValidationResult],
) -> None:
    """Fail the pipeline when a required validation rule fails."""

    failures = [result for result in results if not result.passed]

    if not failures:
        return

    details = "; ".join(f"{result.rule}={result.value}" for result in failures)

    raise DataQualityError(f"RecomX data-quality validation failed: {details}")


def load_interim_tables(
    interim_dir: str | Path = "data/interim",
) -> dict[str, pd.DataFrame]:
    """Load CP-03 Parquet tables required by CP-04."""

    root = Path(interim_dir)

    required_files = {
        "ratings": root / "ratings.parquet",
        "movies": root / "movies.parquet",
        "tags": root / "tags.parquet",
        "genome_scores": (root / "genome_scores.parquet"),
        "genome_tags": (root / "genome_tags.parquet"),
    }

    missing_files = [str(path) for path in required_files.values() if not path.is_file()]

    if missing_files:
        raise FileNotFoundError("Missing CP-03 interim files: " + ", ".join(missing_files))

    return {name: pd.read_parquet(path) for name, path in required_files.items()}


def run_data_quality_validation(
    interim_dir: str | Path = "data/interim",
) -> list[ValidationResult]:
    """Run the complete CP-04 validation suite."""

    tables = load_interim_tables(interim_dir)

    results: list[ValidationResult] = []

    results.extend(validate_ratings(tables["ratings"]))

    results.extend(validate_movies(tables["movies"]))

    results.extend(validate_tags(tables["tags"]))

    results.extend(
        validate_genome(
            tables["genome_scores"],
            tables["genome_tags"],
        )
    )

    results.extend(
        validate_relationships(
            tables["ratings"],
            tables["movies"],
            tables["tags"],
            tables["genome_scores"],
        )
    )

    return results
