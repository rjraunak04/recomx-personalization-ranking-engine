# RecomX — Production Personalization & Ranking Engine

> **End-to-end recommendation system on MovieLens 25M:** chronological data engineering → baselines → collaborative retrieval → hybrid Top-100 retrieval → learning-to-rank → FastAPI serving → observability, monitoring, Docker and CI/CD.

RecomX is built to answer a production recommendation question, not just demonstrate a notebook algorithm:

**Given a user's historical interactions, how do we retrieve a useful candidate set, rank it into a personalized Top-K list, handle cold-start users, evaluate the system without temporal leakage, and expose the result through a testable serving boundary?**

The project deliberately reports both improvements and weaknesses. Popularity is a strong baseline, collaborative gains are small when fallback traffic dominates, the learned ranker improves deeper ordering but not every metric, and the locked test remains isolated from development.

---

## 1. Problem Statement

MovieLens contains millions of explicit user-movie ratings. The engineering task is to turn those historical events into a reproducible recommendation pipeline that can serve deterministic personalized recommendations.

| Component | Definition |
|---|---|
| User | `userId` |
| Item | `movieId` |
| Feedback | explicit rating from 0.5 to 5.0 |
| Positive relevance | `rating >= 4.0` |
| Objective | personalized Top-K recommendation |
| Final serving depth | Top-20 |
| Evaluation | chronological offline ranking evaluation |
| Core metrics | Precision@K, Recall@K, NDCG@K, MAP@K, HitRate@K |
| Cold start | train-only global popularity fallback |

### Why this is harder than cosine similarity

A useful recommender must solve several problems together: extreme user-item sparsity, popularity concentration, temporal leakage, unseen users, already-seen-item filtering, candidate retrieval, ranking, deterministic serving, artifact validation, and production monitoring. RecomX treats these as one system.

---

## 2. End-to-End Architecture

```text
                         OFFLINE TRAINING / EVALUATION

MovieLens 25M
     |
     v
Ingestion -> Schema/Data Validation -> EDA
     |
     v
Chronological Split
     |
     +--------------------+----------------------+------------------+
     |                    |                      |
     v                    v                      v
  TRAIN               VALIDATION            LOCKED TEST
(model signals)     (development only)     (isolated/frozen)
     |
     v
Train-only Signals
     |
     +--> Global Popularity
     +--> Genre / Content Profiles
     +--> Sparse Item-KNN
     +--> 64-factor Latent SVD
     |
     v
Multi-source Candidate Generation
     |
     v
Reciprocal-Rank Fusion (RRF)
     |
     v
Exactly Top-100 Candidates / User
     |
     v
12-feature Ranking Layer
     |
     v
Deterministic Logistic Ranker
     |
     v
Personalized Top-20


                              ONLINE SERVING

Request -> FastAPI -> Validation -> User Routing
                                |
                    +-----------+-----------+
                    |                       |
               Known user              Cold / no candidates
                    |                       |
             ML-ranked Top-20       Train-only popularity
                    |                       |
                    +-----------+-----------+
                                |
                       Seen-item filtering
                                |
                         Deterministic Top-K
                                |
                    +-----------+-----------+
                    |                       |
              Observability             Metrics /
              structured logs           monitoring
                    |                       |
                    +-----------+-----------+
                                |
                         Docker + CI/CD
```

The most important invariant is **temporal leakage control**. Training signals come from historical training data; validation is used for development; train-seen items are filtered before recommendation; and the locked test partition is not used for model selection or serving.

---

## 3. Data Foundation

### Dataset scale

MovieLens 25M ingestion produced:

| Statistic | Value |
|---|---:|
| Ratings | **25,000,095** |
| Users | **162,541** |
| Rated movies | **59,047** |
| Rating range | 0.5–5.0 |
| Interaction sparsity | **~99.74%** |

The data shows a strong long tail and popularity concentration, making a dense user-item representation inappropriate.

### Chronological split

| Partition | Rows |
|---|---:|
| Train | **20,000,076** |
| Validation | **2,500,009** |
| Locked test | **2,500,010** |

The split is deterministic and chronological rather than random because recommendation is a future-facing problem. Random splitting can expose future interaction structure during development.

Generated/raw MovieLens data and model artifacts are intentionally excluded from Git.

---

## 4. Step-by-Step System Evolution

### Stage 1 — Leakage-safe data engineering

The first stage establishes ingestion, schema checks, data validation, exploratory diagnostics, deterministic chronological splitting, and reproducibility hashes.

**Why first?** A sophisticated recommender evaluated on a leaking or unstable split is less useful than a simple model evaluated correctly.

Key guarantees:

- deterministic ingestion and splitting
- explicit data contracts
- chronology-preserving evaluation
- generated artifact isolation
- reproducibility checks
- locked-test separation

### Stage 2 — Baselines before complex models

Two initial recommenders establish the benchmark:

**Global popularity** ranks items using relevant training interactions. It is simple, strong and also becomes the cold-start fallback.

**Genre-aware personalization** builds user genre preferences from relevant historical interactions and combines genre affinity with a small popularity signal.

At K=20, popularity remained stronger in ranking accuracy:

| Model | Precision | Recall | NDCG | MAP | HitRate |
|---|---:|---:|---:|---:|---:|
| Popularity | **0.2254** | **0.0924** | **0.2615** | **0.1449** | **0.7750** |
| Genre-aware | 0.2087 | 0.0798 | 0.2396 | 0.1352 | 0.6708 |

The genre model nevertheless broadened catalogue exposure. This result motivated collaborative retrieval instead of artificially tuning the content model until it appeared superior.

### Stage 3 — Collaborative retrieval

RecomX next learns user-item structure from positive training feedback (`rating >= 4.0`).

The positive sparse matrix contains **137,682 users**, **24,740 items**, and **9,981,093 positive pairs**. Two approaches are implemented:

- **Item-KNN:** sparse cosine-neighbor graph, up to 100 neighbors/item.
- **Latent SVD:** 64-dimensional latent factors using sparse linear algebra.

No dense user-item matrix is materialized.

At K=20:

| Model | Precision | Recall | NDCG | MAP | HitRate |
|---|---:|---:|---:|---:|---:|
| Item-KNN | **0.2264** | **0.0933** | **0.2626** | **0.1454** | **0.7785** |
| Latent SVD | 0.2261 | 0.0930 | 0.2622 | 0.1451 | 0.7773 |
| Popularity | 0.2254 | 0.0924 | 0.2615 | 0.1449 | 0.7750 |
| Genre-aware | 0.2087 | 0.0798 | 0.2396 | 0.1352 | 0.6708 |

The collaborative improvement is deliberately described as **small**. Much of the aggregate population still uses fallback, so the project also evaluates routing/strategy behavior rather than overclaiming the aggregate gain.

Engineering trade-off: the full Item-KNN graph contains **2,432,856 edges** and took roughly **3 h 7 min** to build on the development machine, while the 64-factor SVD fit took roughly **13.34 s**. Expensive collaborative artifacts are therefore cached and reused.

### Stage 4 — Hybrid Top-100 Candidate Retrieval

A production recommender normally separates **retrieval** from **ranking**. RecomX fuses four signals:

```text
Item-KNN -------+
Latent SVD -----+
Genre/Content --+--> Reciprocal-Rank Fusion --> Top-100 candidates
Popularity -----+
```

The first shallow fusion exposed an important failure: Top-20 source lists could not reliably create a deep candidate pool. Deep collaborative/popularity retrieval was introduced and existing expensive artifacts were reused.

Final retrieval result:

| Retrieval diagnostic | Result |
|---|---:|
| Evaluation users | **17,847** |
| Candidates/user | **100** |
| Candidate rows | **1,784,700** |
| Train-seen violations | **0** |
| Recall@20 | **0.1015** |
| Recall@50 | **0.1909** |
| Recall@100 | **0.2749** |

Recall@100 improved from the shallow-fusion result of **0.0969 → 0.2749 (~2.84×)**. Popularity remains a major contributor, especially for cold users; that limitation is retained rather than hidden.

### Stage 5 — ML Learning-to-Rank Model

The Top-100 candidate pool is transformed into a 12-feature ranking table using retrieval provenance plus train-history statistics.

Feature families include:

- retrieval score/rank and inverse rank
- source count
- Item-KNN, SVD, genre and popularity source indicators
- user history count and mean rating
- item interaction count and mean rating

A deterministic pointwise logistic ranker is fitted on a user-level development split. The ML ranker and retrieval-score baseline are evaluated on the **same 3,570 held-out ranking users**.

| Metric @20 | Retrieval baseline | ML ranker |
|---|---:|---:|
| Precision | 0.2320 | **0.2395** |
| Recall | 0.1033 | **0.1053** |
| NDCG | 0.2718 | **0.2762** |
| MAP | 0.1481 | **0.1520** |
| HitRate | **0.8157** | 0.8109 |

NDCG@20 improves by about **0.00438 (~1.61% relative)**. HitRate@20 decreases slightly, so the correct conclusion is that the model improves deeper ordering quality on this development cohort—not that it wins every metric.

### Stage 6 — FastAPI Serving

The offline pipeline is converted into a typed serving boundary with FastAPI.

```text
GET /v1/recommendations/{user_id}?top_k=20
                   |
             request validation
                   |
          +--------+--------+
          |                 |
       known user        unseen user
          |                 |
       ML ranker       popularity fallback
          |                 |
          +--------+--------+
                   |
          deterministic response
```

Serving includes:

- typed request/response contracts
- cached model-artifact loading and feature-schema validation
- known-user ML scoring
- train-seen filtering
- deterministic tie-breaking
- train-only popularity fallback
- health/readiness/metrics endpoints
- explicit 422/503 failure handling without exposing local artifact paths
- reusable movie metadata/explanation utilities

### Stage 7 — Productionization and observability

The final stage adds the engineering boundary around the model:

- environment-driven runtime settings and artifact readiness
- machine-readable, privacy-conscious serving events
- aggregate request/success/error/latency metrics
- personalized vs cold-start traffic monitoring
- feature schema, missingness, standardized mean-shift and score-distribution health checks
- non-root Docker runtime contract
- GitHub Actions quality gates on Python 3.11 and 3.12
- deterministic production smoke/load benchmark

Monitoring components are **health primitives**, not a claim that production drift has already been statistically established; real drift analysis requires live traffic and an approved reference window.

---

## 5. Final System Results

### Retrieval quality

**Hybrid candidate Recall@100: 0.2749**, with exactly 100 candidates for all 17,847 evaluation users and zero train-seen violations.

### Ranking quality

The learned ranker improves Precision@20, Recall@20, NDCG@20 and MAP@20 over the frozen retrieval-score baseline while slightly reducing HitRate@20.

### Serving benchmark

A deterministic 40-request workload mixing personalized and cold-start traffic against the real local backend produced:

| Serving metric | Result |
|---|---:|
| Successful requests | **40 / 40** |
| Errors | **0** |
| Success rate | **100%** |
| Mean latency | **25.69 ms** |
| P95 latency | **41.52 ms** |
| Max latency | **57.56 ms** |
| Throughput | **38.93 req/s** |

These numbers are **local engineering measurements, not a production SLA or capacity claim**.

---

## 6. Evaluation & Leakage Controls

RecomX uses explicit safeguards because recommendation metrics are easy to inflate accidentally:

1. **Chronological split** instead of random future leakage.
2. **Train-only fitting/signals** for recommendation models and fallback.
3. **Validation-only development** for model comparison.
4. **Train-seen filtering** before recommendation.
5. **Deterministic ranking/tie-breaking** for reproducibility.
6. **Warm/cold and strategy diagnostics** instead of aggregate metrics alone.
7. **Locked test isolation** throughout the implemented development and serving workflow.

The locked test is intentionally **not reported as a final performance estimate** in this repository state. This prevents repeated optimization against the final holdout.

---

## 7. Repository Structure

```text
recomx-personalization-ranking-engine/
|
+-- configs/                    # data, evaluation, retrieval, ranking, serving,
|                               # production and monitoring contracts
+-- data/
|   +-- raw/                    # local MovieLens source (ignored)
|   +-- interim/                # normalized intermediate artifacts (ignored)
|   +-- processed/              # temporal partitions (ignored)
+-- docs/                       # architecture, experiment and checkpoint records
+-- reports/                    # generated evaluation/model artifacts (ignored)
+-- scripts/                    # reproducible build/evaluate/verify/benchmark entrypoints
+-- src/recomx/
|   +-- data/                   # ingestion, validation, EDA, splitting, reproducibility
|   +-- evaluation/             # ground truth, metrics, diagnostics, retrieval evaluation
|   +-- models/                 # popularity, content, Item-KNN, SVD, hybrid retrieval
|   +-- ranking/                # ranking contract, dataset, features, ML ranker
|   +-- serving/                # API, artifacts, routing, fallback, metrics, logging
|   +-- monitoring/             # model/data/serving health checks
+-- tests/                      # unit, contract, integration and production checks
+-- .github/workflows/ci.yml    # automated quality contract
+-- Dockerfile                  # non-root serving image contract
+-- pyproject.toml              # package/dependency/tooling configuration
```

The repository keeps reusable implementation under `src/recomx`; scripts orchestrate reproducible workflows; generated datasets/models are not committed.

---

## 8. Reproducibility & Quality Gates

The latest verified local project gate completed with:

- **275 tests passed**
- Ruff: **PASS**
- Python bytecode compilation: **PASS**
- Git diff hygiene: **PASS**
- Day-07 final productionization QA: **PASS**
- working tree clean at verification time

Run the complete local source-quality gate:

```bash
python scripts/run_quality_gates.py
```

Run final productionization checks when approved serving artifacts are present:

```bash
python scripts/run_day07_final_qa.py
python scripts/run_production_smoke_benchmark.py
```

GitHub Actions defines the repository quality contract for Python **3.11 and 3.12**. Generated MovieLens/model artifacts remain outside source control and are not fabricated merely to make CI look green.

---

## 9. Production API Contract

Main endpoints:

```text
GET /health
GET /ready
GET /metrics
GET /v1/recommendations/{user_id}?top_k=20
```

Routing behavior:

```text
known user + available candidates -> ML personalized ranking
unknown/candidate-less user       -> train-only popularity fallback
```

Top-K is capped at 20, ranks are contiguous, movie IDs are unique, ordering is deterministic, and known-user recommendations remove train-seen items.

---

## 10. Key Engineering Decisions

**Why chronological evaluation?**  
Recommendation predicts future preference. Preserving event order reduces leakage from future interactions.

**Why keep a popularity model?**  
It is both a difficult baseline to beat and a deterministic fallback when personalization history is unavailable.

**Why sparse Item-KNN/SVD?**  
The interaction matrix is ~99.74% sparse. Dense reconstruction would waste memory and does not match the data structure.

**Why retrieval + ranking?**  
Retrieval cheaply maximizes candidate coverage; ranking can then apply richer signals to a much smaller set.

**Why Reciprocal-Rank Fusion?**  
The retrieval sources have different score scales. RRF combines ranked evidence without pretending their raw scores are directly comparable.

**Why report regressions?**  
A production model is a trade-off. The ML ranker improves several @20 metrics but slightly lowers HitRate; hiding that would make the evaluation less trustworthy.

**Why isolate the locked test?**  
Repeatedly checking a final holdout turns it into another validation set. The project keeps that boundary explicit.

---

## 11. What This Project Demonstrates

### Data Science

- recommendation problem formulation
- sparse/long-tail EDA
- leakage-safe temporal evaluation
- baseline design
- collaborative and content modeling
- ranking metrics and diagnostics
- retrieval recall analysis
- learning-to-rank feature engineering
- warm/cold-start interpretation
- honest model comparison

### ML Engineering

- modular Python package architecture
- reproducible data/model pipelines
- sparse computation and artifact reuse
- typed serving contracts
- FastAPI serving
- deterministic routing and fallback
- artifact/schema validation
- structured observability and aggregate metrics
- monitoring primitives
- automated tests and quality gates
- Docker/CI deployment contracts

---

## 12. Scope & Honest Limitations

RecomX is deliberately production-oriented, but it does not pretend to be a hyperscale recommender.

- The ranker is a deterministic **pointwise logistic model**, not LambdaMART/listwise/deep ranking.
- Ranking development uses a deterministic user split inside the validation candidate set; a stricter experiment would introduce an additional temporal ranking-training boundary within historical training data.
- Candidate Recall@100 remains an upstream ceiling for ranking quality.
- Popularity contributes heavily to cold-start traffic.
- Current serving uses local precomputed artifacts; a high-QPS deployment would introduce indexed/online candidate storage and stronger caching.
- Monitoring implements health checks; statistically validated drift detection needs real production windows.
- The smoke benchmark is single-environment local evidence, not an SLA.
- The locked test remains intentionally isolated rather than repeatedly consumed during development.

These are explicit boundaries for future work, not hidden gaps.

---

## 13. Project Status

**Day 01–07 complete**

```text
Data foundation
    -> Baselines
    -> Collaborative filtering
    -> Hybrid Top-100 retrieval
    -> Learning-to-rank
    -> Recommendation API
    -> Productionization + observability
```

RecomX is designed to demonstrate the complete reasoning chain expected from a **Data Scientist / ML Engineer**: understand the recommendation problem, establish trustworthy evaluation, build increasingly capable models, measure trade-offs, convert the model into a service, and add the controls required to operate it responsibly.

For deeper implementation and experiment details, see `docs/`.
