# Day-05 — Learning-to-Rank Results

## Objective

Re-rank the frozen Day-04 Top-100 candidate pool into a final personalized Top-20 list without reading the locked test partition.

## Development protocol

The ranking dataset contains 1,784,700 candidate rows for 17,847 validation users. Features are derived from training history and frozen retrieval metadata. For the ML experiment, users are split deterministically: 14,277 users fit the logistic ranker and 3,570 users form the held-out development cohort. The deterministic retrieval baseline and ML ranker are evaluated on exactly the same 3,570 users.

This is a development-validation comparison, not the final locked-test estimate.

## Held-out results

| Model | K | Precision | Recall | NDCG | MAP | Hit Rate |
|---|---:|---:|---:|---:|---:|---:|
| Retrieval baseline | 5 | 0.337199 | 0.040964 | 0.349501 | 0.259178 | 0.699720 |
| ML logistic ranker | 5 | 0.335462 | 0.041042 | 0.348455 | 0.258773 | 0.692717 |
| Retrieval baseline | 10 | 0.278039 | 0.065420 | 0.305955 | 0.194154 | 0.765266 |
| ML logistic ranker | 10 | 0.282689 | 0.066585 | 0.308876 | 0.198279 | 0.752381 |
| Retrieval baseline | 20 | 0.232003 | 0.103269 | 0.271795 | 0.148066 | 0.815686 |
| ML logistic ranker | 20 | 0.239454 | 0.105298 | 0.276173 | 0.152041 | 0.810924 |

NDCG delta (ML minus baseline): -0.001047 @5, +0.002921 @10, and +0.004378 @20.

## Interpretation

The ML ranker provides a modest improvement at K=10 and K=20, including NDCG@20, MAP@20, Recall@20, and Precision@20. It does not dominate every metric: NDCG@5 and Hit Rate are slightly lower. Therefore the correct conclusion is that the ranker improves deeper ordering quality on this held-out development cohort, not that it universally beats the retrieval baseline.

The absolute NDCG@20 gain is about 0.00438 (roughly 1.61% relative). Candidate Recall@100 from Day-04 remains the upstream ceiling, so future ranking gains also depend on stronger retrieval diversity and recall.

## Leakage boundary

- Training-history features only.
- Same held-out users for baseline and ML comparison.
- Locked test not read during Day-05 development.
- Final locked-test evaluation must happen only after model/pipeline freeze.

## Recruiter takeaway

RecomX now separates retrieval from ranking: hybrid retrieval generates a broad Top-100 candidate set, and a learned ranker reorders those candidates into Top-20 recommendations. The benchmark records both improvements and regressions rather than hiding unfavorable metrics.
