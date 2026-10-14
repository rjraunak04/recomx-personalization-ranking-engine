# RecomX Day-04 — Hybrid Candidate Retrieval

Day-04 converts the Day-03 Top-K recommenders into a retrieval layer for a
future learning-to-rank system.

## Contract

- training data only for retrieval features and model fitting
- validation only for offline development evaluation
- locked test remains untouched
- all train-seen items are filtered
- candidate budget: 100 unique items per user where available
- deterministic reciprocal-rank fusion
- collaborative, content and popularity sources
- cold users remain explicitly supported

## Why Hybrid Retrieval?

Day-03 showed that collaborative filtering improves ranking accuracy only
slightly over popularity and remains concentrated on training-popular items.
Day-02 showed that content signals broaden catalogue exposure. Day-04 combines
these complementary signals instead of forcing one model to solve every
retrieval problem.

The retrieval objective is candidate recall and useful candidate diversity,
not final ranking quality. Final ordering is intentionally deferred to a later
ranker.
