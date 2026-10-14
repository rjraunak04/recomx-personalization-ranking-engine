# Day-05 — Learning-to-Rank

## Objective

Day-05 converts the Day-04 Top-100 hybrid retrieval pool into a final personalized Top-20 ranking layer.

## Leakage boundary

- Training-derived features may use chronological training data only.
- Validation is the development/model-selection partition.
- The locked test partition remains inaccessible until the final evaluation stage.
- All train-seen items remain excluded.
- Relevance is frozen at rating >= 4.0.
- Candidate retrieval is frozen to the Day-04 Top-100 interface.

## Checkpoints

1. Ranking contract
2. Ranking dataset
3. Ranking feature engineering
4. Deterministic baseline ranker
5. ML ranking model
6. Final Top-20 generation
7. Offline evaluation and diagnostics
8. Benchmark, QA, documentation, and freeze

## Ranking interface

The ranker receives at most 100 candidate movies per user with retrieval score, rank and source provenance. It returns deterministic final Top-20 recommendations. Ranking metrics remain Precision, Recall, NDCG, MAP and HitRate at K=5/10/20.

Day-04 candidate recall is treated as the retrieval ceiling; Day-05 is responsible for ordering retrieved candidates, not silently expanding the candidate set.
