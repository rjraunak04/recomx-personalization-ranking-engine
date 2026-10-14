from recomx.serving.benchmark import run_backend_benchmark
from recomx.serving.contract import (
    RecommendationItem,
    RecommendationRequest,
    RecommendationResponse,
)


class FastBackend:
    def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
        return RecommendationResponse(
            user_id=request.user_id,
            strategy="personalized",
            recommendations=(
                RecommendationItem(
                    movie_id=10,
                    rank=1,
                    score=0.9,
                    reason="benchmark",
                ),
            ),
        )


class FailingBackend:
    def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
        raise ValueError("synthetic failure")


def test_benchmark_counts_successes() -> None:
    result = run_backend_benchmark(
        FastBackend(),
        tuple(RecommendationRequest(user_id=i, top_k=1) for i in range(1, 6)),
    )
    assert result.requests == 5
    assert result.successes == 5
    assert result.errors == 0
    assert result.success_rate == 1.0
    assert result.throughput_rps > 0


def test_benchmark_counts_controlled_failures() -> None:
    result = run_backend_benchmark(
        FailingBackend(),
        (RecommendationRequest(user_id=1, top_k=1),),
    )
    assert result.successes == 0
    assert result.errors == 1
    assert result.success_rate == 0.0


def test_benchmark_rejects_empty_workload() -> None:
    try:
        run_backend_benchmark(FastBackend(), ())
    except ValueError as exc:
        assert "at least one" in str(exc)
    else:
        raise AssertionError("Empty workload must fail.")
