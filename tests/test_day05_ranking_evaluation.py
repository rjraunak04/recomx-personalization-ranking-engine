from __future__ import annotations

import pandas as pd

from recomx.evaluation.metrics import evaluate_recommendations


def test_ranking_evaluation_rewards_better_ordering() -> None:
    truth = pd.DataFrame({"userId": [1, 1], "movieId": [10, 20]})
    good = pd.DataFrame(
        {
            "userId": [1, 1, 1],
            "movieId": [10, 20, 30],
            "recommendation_rank": [1, 2, 3],
        }
    )
    poor = pd.DataFrame(
        {
            "userId": [1, 1, 1],
            "movieId": [30, 10, 20],
            "recommendation_rank": [1, 2, 3],
        }
    )
    _, good_metrics = evaluate_recommendations(good, truth, cutoffs=(3,))
    _, poor_metrics = evaluate_recommendations(poor, truth, cutoffs=(3,))
    assert good_metrics.loc[0, "ndcg"] > poor_metrics.loc[0, "ndcg"]
    assert good_metrics.loc[0, "map"] > poor_metrics.loc[0, "map"]
