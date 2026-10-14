# RecomX Recommendation Evaluation Protocol

## Purpose

This document defines the offline evaluation contract used by RecomX
recommendation experiments.

The protocol exists to prevent leakage, inconsistent evaluation and accidental
optimization against the locked test set.

## Recommendation Task

The primary task is personalized Top-K recommendation.

For each evaluable user, a recommender produces an ordered list of candidate
items.

Primary cutoffs are:

- K = 5
- K = 10
- K = 20

## Data Partitions

RecomX uses the chronological partitions established during Day 01.

### Train

Used for:

- fitting recommendation models
- estimating item popularity
- constructing user histories
- learning user/item representations in later stages

### Validation

Used for:

- offline model comparison
- model selection
- hyperparameter selection
- development-time ranking evaluation

### Locked Test

Reserved for final evaluation.

The locked test set must not be used for:

- model selection
- hyperparameter tuning
- relevance-threshold selection
- feature decisions
- repeated development feedback

Repeatedly inspecting locked-test performance would effectively turn the test
set into another validation set.

## Relevance

MovieLens contains explicit ratings from 0.5 to 5.0.

Day-02 begins with the following configurable relevance definition:

`rating >= 4.0`

This threshold defines positive future outcomes for Top-K evaluation.

Lower ratings are not automatically treated as positive recommendation
targets.

The threshold is part of the evaluation configuration rather than being
silently embedded throughout modeling code.

## Seen-Item Filtering

Items already observed in a user's training history must be excluded from that
user's recommendation list.

This prevents a model from receiving credit for recommending items the user
has already interacted with during training.

## Temporal Leakage Rule

Recommendation generation may use only information available from the
permitted historical partition.

Validation-period or locked-test information must never influence training
features, popularity statistics, user profiles or candidate scores.

## Cold Start

Evaluation will explicitly distinguish users with training history from users
that first appear in later periods.

Cold-start behaviour must be measured rather than silently dropping all such
users from system-level reporting.

## Ranking Metrics

Later Day-02 checkpoints will implement ranking metrics including:

- Precision@K
- Recall@K
- NDCG@K
- MAP@K
- HitRate@K

Additional diagnostics will measure recommendation coverage, popularity
concentration and long-tail exposure.

## Development Rule

All Day-02 recommender development and model comparison uses the validation
partition.

The locked test remains untouched until a later final-evaluation decision.