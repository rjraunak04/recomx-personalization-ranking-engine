# RecomX Popularity Recommendation Baseline

## Purpose

The popularity recommender provides the first production-style baseline
against which later personalized recommendation models can be compared.

A sophisticated recommender should demonstrate measurable improvement over
a simple popularity strategy.

## Training Signal

The model uses only the chronological training partition.

An interaction contributes to item popularity when:

`rating >= 4.0`

The score for an item is the number of relevant training interactions for
that item.

The number of distinct relevant users is retained as a deterministic
secondary ordering signal.

## Ranking

Items are ordered by:

1. relevant interaction count, descending
2. distinct relevant users, descending
3. movie ID, ascending

This creates deterministic recommendation output.

## Seen-Item Filtering

For a user with training history, every item already observed in that
user's training interactions is removed before Top-K selection.

This includes previously seen items regardless of their historical rating.

The recommender therefore does not recommend an already observed training
item merely because it is globally popular.

## Warm Users

Warm users receive the highest-ranked globally popular items that are not
present in their training history.

The popularity baseline itself does not yet model individual taste.

## Cold Users

A user without training history has no seen items.

Such users receive the global popularity ranking directly.

This provides a deterministic cold-start fallback.

## Leakage Protection

Popularity statistics are learned exclusively from training interactions.

Validation outcomes do not contribute to item scores.

The locked test partition is not read.

Validation cohort information is used only to identify which development
users require recommendations.

## Top-K Output

Day-02 uses recommendation cutoffs:

- Top-5
- Top-10
- Top-20

The baseline stores Top-20 recommendations. Smaller cutoffs can later be
obtained by truncating each user's ranked list.

## Evaluation

CP-04 generates recommendations but does not yet claim ranking performance.

Precision, Recall, NDCG, MAP and HitRate are implemented in a later
checkpoint and evaluated against the CP-03 validation ground truth.