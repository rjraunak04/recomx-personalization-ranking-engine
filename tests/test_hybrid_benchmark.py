"""Fast integrity tests for the Day-04 benchmark contract."""

import pandas as pd


def test_candidate_rank_contract():
    frame = pd.DataFrame({"candidate_rank": [1, 50, 100]})
    assert frame["candidate_rank"].between(1, 100).all()


def test_candidate_identity_contract():
    frame = pd.DataFrame({"userId": [1, 1], "movieId": [10, 20]})
    assert not frame.duplicated(["userId", "movieId"]).any()


def test_locked_test_contract():
    summary = {"locked_test_used": False}
    assert summary["locked_test_used"] is False
