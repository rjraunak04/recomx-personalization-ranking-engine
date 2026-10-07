"""Generate RecomX CP-05 recommendation-specific EDA artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from recomx.data.eda import (
    build_summary,
    item_popularity,
    load_eda_tables,
    monthly_interactions,
    rating_distribution,
    user_activity,
)

REPORT_DIR = Path("reports/eda")
FIGURE_DIR = Path("reports/figures")


def save_activity_distribution(
    values: pd.Series,
    path: Path,
    xlabel: str,
    title: str,
) -> None:
    """Save log-scale activity histogram."""

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(
        values.to_numpy(),
        bins=60,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_rating_distribution(
    distribution: pd.DataFrame,
    path: Path,
) -> None:
    """Save rating distribution figure."""

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        distribution["rating"],
        distribution["interaction_count"],
        width=0.35,
    )

    ax.set_xlabel("Rating")
    ax.set_ylabel("Interactions")
    ax.set_title("MovieLens Rating Distribution")

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_popularity_curve(
    popularity: pd.Series,
    path: Path,
) -> None:
    """Save ranked item-popularity curve."""

    ordered = popularity.sort_values(ascending=False).reset_index(drop=True)

    ranks = range(
        1,
        len(ordered) + 1,
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        ranks,
        ordered.to_numpy(),
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("Item popularity rank")
    ax.set_ylabel("Number of interactions")
    ax.set_title("Item Popularity Long-Tail")

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_temporal_activity(
    monthly: pd.DataFrame,
    path: Path,
) -> None:
    """Save monthly interaction trend."""

    plot_frame = monthly.copy()

    plot_frame["month"] = pd.to_datetime(plot_frame["month"])

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(
        plot_frame["month"],
        plot_frame["interaction_count"],
    )

    ax.set_xlabel("Month")
    ax.set_ylabel("Interactions")
    ax.set_title("Monthly Rating Interaction Volume")

    fig.autofmt_xdate()

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)


def main() -> None:
    """Run CP-05 EDA."""

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading MovieLens interim data...")

    ratings, movies = load_eda_tables()

    print(f"Loaded {len(ratings):,} interactions.")

    print("Calculating recommendation metrics...")

    summary = build_summary(
        ratings,
        movies,
    )

    users = user_activity(ratings)
    items = item_popularity(ratings)

    ratings_dist = rating_distribution(ratings)

    monthly = monthly_interactions(ratings)

    with (REPORT_DIR / "summary.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary.to_dict(),
            handle,
            indent=2,
        )

    users.rename_axis("userId").reset_index().to_csv(
        REPORT_DIR / "user_activity.csv",
        index=False,
    )

    items.rename_axis("movieId").reset_index().to_csv(
        REPORT_DIR / "item_popularity.csv",
        index=False,
    )

    ratings_dist.to_csv(
        REPORT_DIR / "rating_distribution.csv",
        index=False,
    )

    monthly.to_csv(
        REPORT_DIR / "monthly_interactions.csv",
        index=False,
    )

    save_activity_distribution(
        users,
        FIGURE_DIR / "user_activity_distribution.png",
        "Interactions per user",
        "User Activity Distribution",
    )

    save_activity_distribution(
        items,
        FIGURE_DIR / "item_popularity_distribution.png",
        "Interactions per item",
        "Item Popularity Distribution",
    )

    save_rating_distribution(
        ratings_dist,
        FIGURE_DIR / "rating_distribution.png",
    )

    save_popularity_curve(
        items,
        FIGURE_DIR / "item_popularity_long_tail.png",
    )

    save_temporal_activity(
        monthly,
        FIGURE_DIR / "monthly_interactions.png",
    )

    print()
    print("RecomX CP-05 Recommendation EDA")
    print("=" * 62)

    for key, value in summary.to_dict().items():
        if isinstance(value, float):
            print(f"{key:<38} {value:.6f}")
        else:
            print(f"{key:<38} {value}")

    print()
    print("EDA artifacts written to reports/eda/")

    print("Figures written to reports/figures/")

    print()
    print("CP-05 PASSED")


if __name__ == "__main__":
    main()
