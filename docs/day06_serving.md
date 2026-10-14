# Day-06 — Recommendation Serving

## Objective
Day-06 converts RecomX from an offline ranking pipeline into a deterministic, testable recommendation-serving boundary without touching the locked test partition.

## Architecture
Client → FastAPI → request validation → recommendation backend → known/cold routing → ranking or train-only popularity fallback → metadata/explanation enrichment → validated Top-K response.

## Checkpoints
- CP-01: framework-independent serving request/response contract.
- CP-02: validated and cached Day-05 ranking artifact loader.
- CP-03: known-user ML scoring, seen-item filtering, deterministic Top-K.
- CP-04: deterministic train-only popularity fallback for unseen users.
- CP-05: FastAPI health, readiness, and versioned recommendation endpoint.
- CP-06: MovieLens title/genre metadata and strategy-aware explanations.
- CP-07: explicit 422/503 failure semantics and artifact-path protection.
- CP-08: integration benchmark, full QA, documentation, and freeze gate.

## Serving invariants
Top-K is limited to 20. Recommendation ranks are contiguous and movie IDs unique. Known-user ranking uses the frozen Day-05 feature schema and model artifact. Cold-start popularity is fitted from training data only. Personalized serving removes train-seen items. movieId is the deterministic tie-break. Metadata enrichment never changes ranking scores/order. Missing serving artifacts return 503 without leaking local paths. The locked test is not read during Day-06.

## API
- GET /health
- GET /ready
- GET /v1/recommendations/{user_id}?top_k=20

## Final acceptance
Run `python scripts/run_day06_benchmark.py`, then the complete pytest, Ruff, compileall, git diff and git status checks. Day-06 is frozen only after all gates pass.
