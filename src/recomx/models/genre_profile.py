"""Genre-aware personalized recommendation baseline."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from recomx.evaluation.relevance import RelevancePolicy
from recomx.models.popularity import PopularityModel

NO_GENRES = "(no genres listed)"


def _explode_movie_genres(
    movies: pd.DataFrame,
) -> pd.DataFrame:
    """Convert MovieLens pipe-separated genres to movie-genre rows."""

    required = {"movieId", "genres"}
    missing = required.difference(movies.columns)

    if missing:
        raise ValueError(f"movies missing required columns: {sorted(missing)}")

    if movies.empty:
        raise ValueError("movies cannot be empty.")

    frame = movies[["movieId", "genres"]].copy()

    frame["genres"] = frame["genres"].fillna(NO_GENRES).astype(str).str.split("|")

    frame = frame.explode(
        "genres",
        ignore_index=True,
    )

    frame["genres"] = frame["genres"].str.strip()

    frame = frame.loc[frame["genres"].ne("") & frame["genres"].ne(NO_GENRES)]

    return frame.drop_duplicates(["movieId", "genres"]).reset_index(drop=True)


@dataclass(frozen=True)
class GenreProfileModel:
    """Personalized genre-affinity model with popularity fallback."""

    movie_genres: pd.DataFrame
    user_genre_profiles: pd.DataFrame
    popularity_model: PopularityModel
    popularity_weight: float = 0.05

    @classmethod
    def fit(
        cls,
        train: pd.DataFrame,
        movies: pd.DataFrame,
        policy: RelevancePolicy | None = None,
        popularity_weight: float = 0.05,
    ) -> GenreProfileModel:
        """Fit user genre preferences from relevant training history."""

        required = {"userId", "movieId", "rating"}
        missing = required.difference(train.columns)

        if missing:
            raise ValueError(f"train missing required columns: {sorted(missing)}")

        if train.empty:
            raise ValueError("train cannot be empty.")

        if not 0.0 <= popularity_weight <= 1.0:
            raise ValueError("popularity_weight must be between 0 and 1.")

        active_policy = policy or RelevancePolicy()
        active_policy.validate()

        relevant = active_policy.relevant_interactions(train)

        movie_genres = _explode_movie_genres(movies)

        history = relevant[["userId", "movieId"]].merge(
            movie_genres,
            on="movieId",
            how="inner",
            validate="many_to_many",
        )

        genre_counts = (
            history.groupby(
                ["userId", "genres"],
                as_index=False,
            )
            .size()
            .rename(columns={"size": "genre_count"})
        )

        totals = genre_counts.groupby("userId")["genre_count"].transform("sum")

        genre_counts["genre_affinity"] = genre_counts["genre_count"] / totals

        popularity_model = PopularityModel.fit(
            train=train,
            policy=active_policy,
        )

        return cls(
            movie_genres=movie_genres,
            user_genre_profiles=genre_counts,
            popularity_model=popularity_model,
            popularity_weight=popularity_weight,
        )

    def has_profile(
        self,
        user_id: int,
    ) -> bool:
        """Return whether a user has a learned genre profile."""

        return bool(self.user_genre_profiles["userId"].eq(user_id).any())

    def _normalized_popularity(
        self,
    ) -> pd.DataFrame:
        """Return item popularity scaled into [0, 1]."""

        popularity = self.popularity_model.item_scores[["movieId", "popularity_score"]].copy()

        maximum = popularity["popularity_score"].max()

        if maximum <= 0:
            popularity["normalized_popularity"] = 0.0
        else:
            popularity["normalized_popularity"] = popularity["popularity_score"] / maximum

        return popularity[
            [
                "movieId",
                "normalized_popularity",
            ]
        ]

    def score_items_for_user(
        self,
        user_id: int,
    ) -> pd.DataFrame:
        """Score catalogue items for a warm profiled user."""

        profile = self.user_genre_profiles.loc[
            self.user_genre_profiles["userId"].eq(user_id),
            ["genres", "genre_affinity"],
        ]

        if profile.empty:
            return pd.DataFrame(
                columns=[
                    "movieId",
                    "genre_score",
                    "normalized_popularity",
                    "final_score",
                ]
            )

        scored_genres = self.movie_genres.merge(
            profile,
            on="genres",
            how="inner",
            validate="many_to_many",
        )

        genre_scores = (
            scored_genres.groupby(
                "movieId",
                as_index=False,
            )["genre_affinity"]
            .sum()
            .rename(columns={"genre_affinity": "genre_score"})
        )

        popularity = self._normalized_popularity()

        candidates = genre_scores.merge(
            popularity,
            on="movieId",
            how="left",
            validate="one_to_one",
        )

        candidates["normalized_popularity"] = candidates["normalized_popularity"].fillna(0.0)

        candidates["final_score"] = (1.0 - self.popularity_weight) * candidates[
            "genre_score"
        ] + self.popularity_weight * candidates["normalized_popularity"]

        return candidates

    def recommend(
        self,
        user_id: int,
        seen_items: set[int] | None = None,
        k: int = 20,
    ) -> pd.DataFrame:
        """Generate deterministic personalized Top-K recommendations."""

        if k <= 0:
            raise ValueError("k must be positive.")

        seen = seen_items or set()

        if not self.has_profile(user_id):
            fallback = self.popularity_model.recommend(
                user_id=user_id,
                seen_items=seen,
                k=k,
            )

            fallback = fallback[
                [
                    "userId",
                    "movieId",
                    "popularity_score",
                    "recommendation_rank",
                ]
            ].copy()

            fallback["genre_score"] = 0.0
            fallback["final_score"] = fallback["popularity_score"].astype(float)
            fallback["strategy"] = "popularity_fallback"

            return fallback[
                [
                    "userId",
                    "movieId",
                    "genre_score",
                    "popularity_score",
                    "final_score",
                    "recommendation_rank",
                    "strategy",
                ]
            ]

        candidates = self.score_items_for_user(user_id)

        candidates = candidates.loc[~candidates["movieId"].isin(seen)].copy()

        candidates = candidates.merge(
            self.popularity_model.item_scores[["movieId", "popularity_score"]],
            on="movieId",
            how="left",
            validate="one_to_one",
        )

        candidates["popularity_score"] = candidates["popularity_score"].fillna(0).astype(int)

        candidates = candidates.sort_values(
            [
                "final_score",
                "popularity_score",
                "movieId",
            ],
            ascending=[
                False,
                False,
                True,
            ],
            kind="mergesort",
        ).head(k)

        output = candidates[
            [
                "movieId",
                "genre_score",
                "popularity_score",
                "final_score",
            ]
        ].copy()

        output.insert(
            0,
            "userId",
            user_id,
        )

        output["recommendation_rank"] = range(
            1,
            len(output) + 1,
        )

        output["strategy"] = "genre_personalized"

        return output.reset_index(drop=True)


def recommend_genre_for_users(
    model: GenreProfileModel,
    user_ids: list[int],
    seen_item_map: dict[int, set[int]],
    k: int = 20,
) -> pd.DataFrame:
    """Generate genre-aware recommendations for multiple users."""

    outputs: list[pd.DataFrame] = []

    for user_id in user_ids:
        outputs.append(
            model.recommend(
                user_id=int(user_id),
                seen_items=seen_item_map.get(
                    int(user_id),
                    set(),
                ),
                k=k,
            )
        )

    if not outputs:
        return pd.DataFrame(
            columns=[
                "userId",
                "movieId",
                "genre_score",
                "popularity_score",
                "final_score",
                "recommendation_rank",
                "strategy",
            ]
        )

    return pd.concat(
        outputs,
        ignore_index=True,
    )
