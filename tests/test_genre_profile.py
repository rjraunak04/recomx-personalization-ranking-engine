"""Tests for genre-aware personalized recommendations."""

import pandas as pd
import pytest

from recomx.models.genre_profile import (
    GenreProfileModel,
    _explode_movie_genres,
    recommend_genre_for_users,
)
from recomx.models.popularity import (
    build_seen_item_map,
)


@pytest.fixture
def movies() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "movieId": [
                10,
                20,
                30,
                40,
                50,
                60,
            ],
            "title": [
                "A",
                "B",
                "C",
                "D",
                "E",
                "F",
            ],
            "genres": [
                "Action|Adventure",
                "Drama",
                "Action",
                "Comedy",
                "Drama|Romance",
                "(no genres listed)",
            ],
        }
    )


@pytest.fixture
def train() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "userId": [
                1,
                1,
                2,
                2,
                3,
                4,
            ],
            "movieId": [
                10,
                20,
                20,
                50,
                30,
                40,
            ],
            "rating": [
                5.0,
                2.0,
                5.0,
                4.5,
                4.0,
                2.0,
            ],
        }
    )


def test_movie_genres_are_exploded(
    movies: pd.DataFrame,
) -> None:
    exploded = _explode_movie_genres(movies)

    genres = set(
        exploded.loc[
            exploded["movieId"] == 10,
            "genres",
        ]
    )

    assert genres == {
        "Action",
        "Adventure",
    }


def test_no_genres_listed_is_excluded(
    movies: pd.DataFrame,
) -> None:
    exploded = _explode_movie_genres(movies)

    assert 60 not in set(exploded["movieId"])


def test_user_profile_uses_relevant_history_only(
    train: pd.DataFrame,
    movies: pd.DataFrame,
) -> None:
    model = GenreProfileModel.fit(
        train,
        movies,
    )

    profile = model.user_genre_profiles.loc[model.user_genre_profiles["userId"].eq(1)]

    assert set(profile["genres"]) == {
        "Action",
        "Adventure",
    }


def test_genre_affinity_sums_to_one_per_user(
    train: pd.DataFrame,
    movies: pd.DataFrame,
) -> None:
    model = GenreProfileModel.fit(
        train,
        movies,
    )

    totals = model.user_genre_profiles.groupby("userId")["genre_affinity"].sum()

    assert all(value == pytest.approx(1.0) for value in totals)


def test_personalized_user_prefers_matching_genre(
    train: pd.DataFrame,
    movies: pd.DataFrame,
) -> None:
    model = GenreProfileModel.fit(
        train,
        movies,
        popularity_weight=0.0,
    )

    recommendations = model.recommend(
        user_id=1,
        seen_items={10, 20},
        k=3,
    )

    assert recommendations.iloc[0]["movieId"] == 30


def test_seen_items_are_filtered(
    train: pd.DataFrame,
    movies: pd.DataFrame,
) -> None:
    model = GenreProfileModel.fit(
        train,
        movies,
    )

    recommendations = model.recommend(
        user_id=2,
        seen_items={20, 50},
        k=5,
    )

    assert not set(recommendations["movieId"]).intersection({20, 50})


def test_cold_user_uses_popularity_fallback(
    train: pd.DataFrame,
    movies: pd.DataFrame,
) -> None:
    model = GenreProfileModel.fit(
        train,
        movies,
    )

    recommendations = model.recommend(
        user_id=999,
        seen_items=set(),
        k=3,
    )

    assert (recommendations["strategy"] == "popularity_fallback").all()


def test_user_without_positive_profile_uses_fallback(
    train: pd.DataFrame,
    movies: pd.DataFrame,
) -> None:
    model = GenreProfileModel.fit(
        train,
        movies,
    )

    assert not model.has_profile(4)

    recommendations = model.recommend(
        user_id=4,
        seen_items={40},
        k=2,
    )

    assert (recommendations["strategy"] == "popularity_fallback").all()


def test_batch_recommendation_generation(
    train: pd.DataFrame,
    movies: pd.DataFrame,
) -> None:
    model = GenreProfileModel.fit(
        train,
        movies,
    )

    seen = build_seen_item_map(train)

    recommendations = recommend_genre_for_users(
        model=model,
        user_ids=[1, 999],
        seen_item_map=seen,
        k=2,
    )

    assert set(recommendations["userId"]) == {1, 999}


def test_invalid_popularity_weight_fails(
    train: pd.DataFrame,
    movies: pd.DataFrame,
) -> None:
    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        GenreProfileModel.fit(
            train,
            movies,
            popularity_weight=1.5,
        )


def test_invalid_k_fails(
    train: pd.DataFrame,
    movies: pd.DataFrame,
) -> None:
    model = GenreProfileModel.fit(
        train,
        movies,
    )

    with pytest.raises(
        ValueError,
        match="positive",
    ):
        model.recommend(
            user_id=1,
            k=0,
        )
