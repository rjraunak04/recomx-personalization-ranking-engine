@'
# RecomX — Personalization & Ranking Engine

Production-oriented recommendation and personalization system built around
large-scale interaction data, leakage-safe evaluation and reproducible ML
engineering.

RecomX is being developed incrementally from a validated recommendation-data
foundation toward candidate retrieval, personalized ranking, cold-start
handling, APIs, monitoring and deployment.

## Current Status

**Day 01 — Data Foundation: Complete**

The current repository establishes the data and evaluation foundation required
before recommendation models are introduced.

No collaborative-filtering, embedding or ranking model is intentionally trained
during Day 01.

## Dataset

RecomX currently uses **MovieLens 25M**.

The source contains approximately 25 million explicit movie-rating interactions
together with movie metadata, user-generated tags, links and genome features.

Primary recommendation entities:

- User: `userId`
- Item: `movieId`
- Explicit feedback: `rating`
- Event time: `timestamp`

Raw and generated datasets are excluded from Git.

## Day-01 Pipeline

```text
MovieLens 25M
      |
      v
Schema-Aware Ingestion
      |
      v
Recommendation Data Validation
      |
      v
Recommendation-Specific EDA
      |
      v
Leakage-Safe Temporal Split
      |
      v
Deterministic Reproducibility Audit
## Day-02 � Recommendation Baselines and Offline Evaluation

RecomX now includes a leakage-safe Top-K recommendation benchmark built on the
MovieLens 25M temporal split.

Implemented components:

- explicit-feedback relevance policy (`rating >= 4.0`)
- leakage-safe validation ground truth
- warm/cold user cohorts
- train-only popularity recommender
- genre-aware personalized recommender
- popularity fallback for cold users
- seen-item filtering
- Precision@K, Recall@K, NDCG@K, MAP@K and HitRate@K
- catalogue and eligible-item coverage
- popularity-bias and long-tail diagnostics
- cold-item exposure analysis
- strategy-level and cohort-level evaluation

The Day-02 benchmark evaluates 17,847 users against 1,231,511 relevant
validation interactions.

At K=10, the popularity baseline achieved NDCG 0.294354 and HitRate 0.719897,
outperforming the simple genre-aware baseline on ranking accuracy. The
genre-aware model provided broader catalogue and long-tail exposure, creating
a clear accuracy-diversity trade-off for later collaborative and hybrid
models to improve.

The locked test partition remains untouched.
