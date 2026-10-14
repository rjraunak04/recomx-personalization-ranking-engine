# Day-07 — Productionization and Observability

## CP-01: Production runtime contract

RecomX now has an explicit environment-driven serving runtime contract. Runtime configuration separates deployment concerns from recommendation logic and validates environment, host, port, required artifact locations, readiness, and locked-test isolation.

The production runtime requires the Day-05 ranking model, ranking features, chronological training ratings, and MovieLens metadata. Readiness exposes artifact availability by filename only so local filesystem roots are not leaked.

The locked test partition is forbidden from production-serving configuration. Day-07 continues to use only already-approved development/serving artifacts.


## CP-02: Structured serving observability

Recommendation requests now emit machine-readable JSON events with request outcome, latency, serving strategy, recommendation count, and safe error type. The event schema intentionally excludes user IDs and filesystem paths. Both successful and handled failure paths are instrumented at the API boundary so serving behavior can be aggregated later without coupling monitoring to ranking logic.


## CP-03: Recommendation metrics monitoring

The API now aggregates privacy-safe serving metrics in process: request volume, success/error rates, personalized versus cold-start mix, recommendation counts, and mean/max latency. Metrics are updated from the same structured serving events used by CP-02, avoiding duplicated instrumentation logic. A lightweight `/metrics` snapshot endpoint exposes aggregate values only; no user identifiers or filesystem paths are retained.


## CP-04: Model and data monitoring

RecomX now includes explicit serving-health checks for feature schema quality, missing-value rate, standardized feature-mean shift, model probability validity and score distribution, plus cold-start traffic share. Thresholds are documented in `configs/monitoring.yaml`. The checks return compact alert codes suitable for dashboards or alerting without retaining user-level data.

These checks are monitoring primitives rather than a claim of statistically confirmed production drift: real drift conclusions require live serving windows and an approved reference window. The locked test partition remains excluded.


## CP-05: Container and deployment readiness

RecomX now has a concrete production backend instead of a benchmark-only API adapter. Startup validates required artifacts, known users route through the Day-05 ranker when serving candidates exist, unknown users and known users without precomputed candidates use the deterministic train-only popularity fallback, and `/ready` is wired to real artifact readiness.

The container runs as a non-root user and starts `recomx.serving.production:app` with Uvicorn. Large generated artifacts remain outside Git; a deployable image therefore requires the approved local serving artifacts to be present in the Docker build context. The locked test partition is neither configured nor copied.
