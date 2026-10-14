# RecomX — Personalization & Ranking Engine

Production-oriented recommendation and personalization system built around large-scale interaction data, leakage-safe evaluation, and reproducible ML engineering.

RecomX is developed incrementally from a validated recommendation-data foundation toward candidate retrieval, personalized ranking, cold-start handling, APIs, monitoring, and deployment.

## Current Status

**Day 01 — Data Foundation: Complete**

**Day 02 — Recommendation Baselines and Offline Evaluation: Complete**

**Day 03 — Collaborative Filtering and Latent-Factor Retrieval: In Progress**

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
