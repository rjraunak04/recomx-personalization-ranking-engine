# Validation Ground Truth and Evaluation Cohorts

## Purpose

RecomX evaluates recommendation quality against future relevant
interactions from the chronological validation partition.

The training partition provides historical information. The validation
partition provides development-time future outcomes.

The locked test partition is not used during this process.

## Relevance

A validation interaction is relevant when:

`rating >= 4.0`

The threshold comes from the centralized RecomX relevance policy.

## Seen-Item Filtering

For a warm user, an item already present in that user's training history
is not a valid new recommendation target.

Therefore relevant validation interactions that duplicate a user's
training user-item pair are removed from ground truth.

This aligns target construction with the candidate-generation rule that
previously seen training items must not be recommended.

## User Cohorts

### Warm User

A validation user who exists in the training partition.

The system has historical interaction information for this user.

### Cold User

A validation user who does not exist in the training partition.

The system has no training interaction history for this user.

### Evaluable User

A user with at least one relevant validation target remaining after the
evaluation policy and seen-item filtering are applied.

### User Without Relevant Targets

A validation user with no rating meeting the relevance threshold.

Such a user cannot contribute to standard relevance-based Top-K metrics.

A warm user whose only relevant validation items were already seen during
training may also become non-evaluable after seen-item filtering.

## Leakage Protection

Ground-truth construction reads only:

- training interactions
- validation interactions

It does not read the locked test partition.

Training history determines seen items and warm/cold status.

Validation outcomes determine development-time ground truth.

## Generated Artifacts

The build process creates:

- validation ground-truth interactions
- validation user cohorts
- ground-truth summary statistics

These are generated analytical artifacts and are excluded from version
control.

## Role in Later Checkpoints

The ground truth produced here becomes the reference set for Day-02
ranking metrics and baseline model comparison.

Warm and cold users remain explicitly identifiable so recommendation
performance can later be reported by segment.