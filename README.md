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