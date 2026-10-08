# RecomX Top-K Ranking Metrics

## Purpose

RecomX evaluates recommendation models as ranked Top-K systems rather than
using regression metrics such as RMSE as the primary recommendation metric.

The same validation ground truth and the same evaluation cutoffs are used for
all Day-02 baselines.

## Evaluation Population

Only users with at least one evaluable relevant validation target from the
CP-03 ground-truth pipeline are included.

Relevant validation targets satisfy the explicit-feedback relevance policy:

`rating >= 4.0`

Training-seen relevant targets are removed by the ground-truth pipeline before
evaluation.

## Precision@K

Precision@K measures the fraction of the Top-K recommendation slots occupied
by relevant items.

`Precision@K = relevant recommended items in Top-K / K`

The denominator remains K even if a model returns fewer than K items. This
prevents short recommendation lists from receiving an artificial advantage.

## Recall@K

Recall@K measures the fraction of a user's relevant validation targets that
appear in the Top-K recommendation list.

`Recall@K = relevant recommended items in Top-K / relevant validation items`

## HitRate@K

HitRate@K equals 1 for a user when at least one relevant target appears in the
Top-K list and 0 otherwise.

The reported value is the macro average across evaluation users.

## NDCG@K

NDCG rewards relevant recommendations more strongly when they occur near the
top of the ranked list.

Binary relevance is used because the Day-02 ground truth is defined by the
explicit relevance threshold.

## MAP@K

Average Precision@K computes precision at each relevant hit position and
normalizes by:

`min(number of relevant targets, K)`

MAP@K is the macro average of user-level Average Precision@K.

## Macro Averaging

Metrics are calculated independently for each user and then averaged across
users.

This prevents heavy users from dominating evaluation simply because they have
more interactions.

## Cutoffs

Day-02 evaluates:

- K = 5
- K = 10
- K = 20

## Fair Model Comparison

The popularity and genre-aware models are evaluated against exactly the same
ground truth and user population.

No validation labels are used to fit either baseline.

The locked test partition remains untouched.

## Interpretation

Higher values are better for all five ranking metrics.

However, ranking accuracy alone is not sufficient for a production
recommender. Coverage, popularity bias, long-tail exposure and cold-start
behaviour are evaluated separately in the following checkpoint.