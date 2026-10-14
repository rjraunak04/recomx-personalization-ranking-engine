# RecomX — Personalization & Ranking Engine

Production-oriented recommendation and personalization system built around large-scale interaction data, leakage-safe evaluation, and reproducible ML engineering.

RecomX is developed incrementally from a validated recommendation-data foundation toward candidate retrieval, personalized ranking, cold-start handling, APIs, monitoring, and deployment.

## Current Status

**Day 01 — Data Foundation: Complete**

**Day 02 — Recommendation Baselines and Offline Evaluation: Complete**

**Day 03 — Collaborative Filtering and Latent-Factor Retrieval: Complete**

## Dataset

RecomX uses **MovieLens 25M** with approximately 25 million explicit movie-rating interactions plus movie metadata, user-generated tags, links, and genome features.

Primary recommendation entities:

- User: `userId`
- Item: `movieId`
- Explicit feedback: `rating`
- Event time: `timestamp`

Raw and generated datasets are excluded from Git.

## Day-01 Data Foundation

MovieLens 25M is ingested, validated, profiled, split chronologically, and audited for deterministic reproducibility. The locked test partition is isolated from development.

## Day-02 Recommendation Baselines

Implemented components include:

- explicit-feedback relevance policy (`rating >= 4.0`)
- leakage-safe validation ground truth
- warm/cold user cohorts
- train-only popularity recommender
- genre-aware personalized recommender
- popularity fallback for cold users
- seen-item filtering
- Precision@K, Recall@K, NDCG@K, MAP@K, and HitRate@K
- catalogue, long-tail, popularity-bias, and cold-item diagnostics

The Day-02 benchmark evaluates 17,847 users against 1,231,511 relevant validation interactions.

At K=10, popularity achieved NDCG 0.294354 and HitRate 0.719897, outperforming the simple genre-aware baseline on ranking accuracy. The genre model produced broader catalogue and long-tail exposure, establishing an accuracy-diversity trade-off for later collaborative and hybrid models.

## Day-03 Collaborative Filtering

Day-03 preserves the Day-02 evaluation boundary while adding scalable collaborative retrieval.

Current contract:

- fit only on chronological training data
- positive collaborative feedback is `rating >= 4.0`
- deterministic sparse CSR representation with float32 values
- dense user-item matrices are forbidden
- validation is used for development/model selection
- locked test remains untouched
- all train-seen items are filtered before recommendation
- cold users use the train-only popularity fallback
- evaluate at K = 5, 10, and 20

The sparse interaction layer is the shared foundation for Item-KNN and latent-factor retrieval.

Implemented Day-03 components include sparse positive-interaction modeling, cosine Item-KNN, 64-factor truncated SVD, train-seen filtering, popularity fallback, collaborative candidate generation, frozen-protocol offline evaluation, and warm/cold plus strategy-level diagnostics.

At K=20, Item-KNN achieved NDCG 0.262566 and HitRate 0.778450, narrowly outperforming latent SVD (NDCG 0.262225) and the popularity baseline (NDCG 0.261514). The gain is intentionally reported as small because most evaluation users are routed through popularity fallback.

The strategy-level analysis also shows that SVD performs strongly among users actually served by its collaborative route, while both collaborative systems remain concentrated on popular training items. These findings motivate a later hybrid retrieval and ranking stage rather than overstating Day-03 accuracy gains.

See `docs/day03_collaborative_filtering_results.md` for the full benchmark and engineering interpretation.


## Day-04 Hybrid Candidate Retrieval

Day-04 adds a retrieval layer designed for downstream ranking rather than
treating each recommender's Top-K output as the final architecture.

The hybrid layer combines Item-KNN, latent SVD, genre/content and popularity
sources with deterministic reciprocal-rank fusion, removes train-seen items,
deduplicates candidates, preserves source provenance and targets up to 100
candidates per user. Candidate Recall@20/50/100 and source contribution are
evaluated on the frozen validation ground truth while the locked test remains
untouched.

This creates the interface required for a later learning-to-rank stage:
retrieval produces a broad candidate pool; ranking decides the final order.
