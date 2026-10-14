# RecomX — Production Personalization & Ranking Engine

A recruiter-ready recommendation system built on **MovieLens 25M** that progresses from leakage-safe data engineering to multi-source retrieval, learning-to-rank, production serving, monitoring, CI/CD, and deployment readiness.

## Why RecomX

A recommender is not just a similarity function. RecomX models the full path from historical interactions to a production-style Top-K service:

```text
MovieLens 25M
    |
    v
Validation + Chronological Split
    |
    v
Train-only Recommendation Signals
    |
    +--> Popularity --------------------+
    +--> Genre / Content ---------------+
    +--> Item-KNN ----------------------+
    +--> Latent SVD --------------------+
                                        |
                                        v
                         Hybrid Top-100 Candidate Retrieval
                                        |
                                        v
                              Ranking Feature Layer
                                        |
                                        v
                           ML Learning-to-Rank Model
                                        |
                         +--------------+--------------+
                         |                             |
                    Known User                    Cold Start
                         |                             |
                    ML Top-20                 Train-only Popularity
                         +--------------+--------------+
                                        |
                                        v
                                FastAPI Serving
                                        |
                       +----------------+----------------+
                       |                                 |
                 Observability                     Monitoring
                       |                                 |
                       +----------------+----------------+
                                        |
                                  Docker + CI/CD
```

The central engineering constraint is **temporal leakage control**: training fits only historical data, validation drives development, train-seen items are filtered before recommendation, and the locked test partition is kept outside development and serving.

## Problem Formulation

- **User:** `userId`
- **Item:** `movieId`
- **Feedback:** explicit 0.5–5.0 rating
- **Relevance policy:** `rating >= 4.0`
- **Task:** personalized Top-K recommendation
- **Evaluation:** chronological offline ranking evaluation
- **Primary metrics:** Precision@K, Recall@K, NDCG@K, MAP@K, HitRate@K
- **Serving output:** Top-20 recommendations with deterministic ordering
- **Cold start:** train-only global popularity fallback

## Dataset Scale

MovieLens 25M ingestion contains **25,000,095 ratings**, **162,541 users**, and **59,047 rated movies**. The interaction matrix is extremely sparse (~99.74%), with a strong long-tail/popularity concentration. Raw and generated data are intentionally excluded from Git.

Chronological split:

| Partition | Rows |
|---|---:|
| Train | 20,000,076 |
| Validation | 2,500,009 |
| Locked test | 2,500,010 |

## System Evolution & Results

| Stage | Engineering outcome |
|---|---|
| Data foundation | deterministic ingestion, validation, profiling, temporal split, reproducibility hashes |
| Baselines | popularity + genre-aware recommenders with warm/cold diagnostics |
| Collaborative retrieval | sparse Item-KNN + 64-factor SVD without dense user-item matrices |
| Hybrid retrieval | four-source reciprocal-rank fusion, exactly 100 candidates/user |
| Learning-to-rank | 12-feature deterministic logistic ranker for final Top-20 |
| Serving | FastAPI contract, known/cold routing, artifact validation, failure handling |
| Productionization | structured logs, aggregate metrics, model/data health checks, Docker, CI/CD |
| Performance | 40/40 production-backend requests succeeded; mean 25.77 ms, P95 43.82 ms, 38.81 req/s on the measured local environment |

### Retrieval

The final hybrid candidate layer serves **17,847 validation users** with exactly **100 candidates per user** and zero train-seen leakage. Candidate Recall@100 improved from the earlier shallow fusion result of 0.0969 to **0.2749** after deep source retrieval was introduced.

### Ranking

On the held-out ranking-user subset (3,570 users), the ML ranker reached:

| Metric @20 | Hybrid-score baseline | ML ranker |
|---|---:|---:|
| Precision | 0.2320 | **0.2395** |
| Recall | 0.1033 | **0.1053** |
| NDCG | 0.2718 | **0.2762** |
| MAP | 0.1481 | **0.1520** |
| HitRate | **0.8157** | 0.8109 |

The model improves deeper ordering quality while slightly reducing HitRate@20. This trade-off is reported directly rather than presenting the ranker as universally superior.

### Production smoke benchmark

A deterministic 40-request mixed known/cold workload against the real backend produced:

- **100% success rate (40/40)**
- **0 errors**
- **25.77 ms mean latency**
- **43.82 ms P95 latency**
- **58.09 ms max latency**
- **38.81 requests/sec**

These are local engineering measurements, **not a production SLA**.

## Production Serving

The serving layer includes:

- typed recommendation request/response contracts
- cached ranking-model artifact loading with schema validation
- ML ranking for users with serving candidates
- deterministic popularity fallback for unseen or candidate-less users
- seen-item filtering
- `/health`, artifact-backed `/ready`, and aggregate `/metrics`
- sanitized 422/503 failure behavior
- machine-readable privacy-conscious serving events
- model/data health monitoring primitives
- non-root Docker runtime
- GitHub Actions quality gates for Python 3.11 and 3.12

## Repository Layout

```text
configs/             model, retrieval, ranking, serving, production, monitoring contracts
docs/                checkpoint architecture and benchmark documentation
scripts/             reproducible build, evaluation, verification and benchmark commands
src/recomx/
  data/               ingestion, validation, temporal splitting
  evaluation/         ranking and retrieval evaluation
  models/             baselines, collaborative, hybrid retrieval
  ranking/            ranking dataset, features and ML ranker
  serving/            API, artifacts, routing, metrics, observability, benchmark
  monitoring/         feature/model/serving health checks
tests/                unit, contract, serving and production-quality tests
.github/workflows/    CI quality gates
Dockerfile            non-root production container contract
```

## Quality & Reproducibility

The current local quality gate passes **270 tests**, Ruff, bytecode compilation, and diff hygiene. CI runs the source-code quality contract on Python 3.11/3.12. Generated MovieLens and model artifacts are not committed or fabricated in CI.

Run the local quality contract:

```bash
python scripts/run_quality_gates.py
```

Run the production smoke benchmark after approved serving artifacts are available:

```bash
python scripts/run_production_smoke_benchmark.py
```

## Engineering Decisions

**Why chronological splitting?** Recommendations are future-facing; random splits leak future interaction structure.

**Why sparse collaborative models?** The matrix is ~99.74% sparse, so a dense representation is both wasteful and unnecessary.

**Why retrieval + ranking?** Retrieval optimizes candidate coverage at manageable cost; ranking then uses richer signals to order a small candidate set.

**Why keep popularity?** It is a strong baseline and a deterministic cold-start fallback. Its strength is measured rather than hidden.

**Why a locked test?** Development decisions should not repeatedly optimize against the final evaluation partition. RecomX keeps it isolated throughout the implemented development and serving workflow.

## Scope & Honest Limitations

The current ML ranker is a deterministic pointwise logistic model rather than LambdaMART/listwise ranking. Ranking development uses a deterministic user split inside the validation candidate set; a stricter future experiment would construct an additional temporal ranking-training boundary inside the historical training period. Monitoring provides production-ready primitives, but real drift conclusions require live traffic and approved reference windows. The benchmark measures backend behavior on one local environment and is not a capacity or SLA claim.

## Project Status

**Day 01–07 complete:** data foundation → baselines → collaborative retrieval → hybrid retrieval → learning-to-rank → serving API → productionization/observability.

The repository is intentionally designed to demonstrate both **Data Science reasoning** (evaluation, baselines, ranking metrics, bias/cold-start interpretation) and **ML Engineering** (artifact contracts, APIs, observability, monitoring, Docker, CI/CD, reproducibility).

See `docs/` for checkpoint-level design and benchmark details.
