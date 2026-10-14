# RecomX Genre-Aware Personalized Baseline

## Purpose

The genre-aware baseline introduces user-specific recommendation behaviour
while remaining interpretable and reproducible.

It provides an intermediate baseline between global popularity and later
collaborative-filtering or representation-learning models.

## User Preference Signal

User profiles are learned exclusively from relevant training interactions:

`rating >= 4.0`

For each user, genres associated with relevant historical movies are counted
and normalized into a genre-affinity distribution.

A user who repeatedly gives relevant ratings to Action and Adventure movies,
for example, receives higher affinity for those genres.

## Candidate Scoring

For each candidate movie, genre affinity is calculated from the user's learned
genre preferences.

A small normalized popularity component is included as a secondary signal:

`final_score = 0.95 * genre_score + 0.05 * normalized_popularity`

This gives personalization dominant influence while allowing popularity to
provide a stable secondary signal.

The weight is deliberately simple for the Day-02 baseline and is not tuned
against the locked test set.

## Seen-Item Filtering

All items present in a user's training history are removed before Top-K
selection.

This applies regardless of the historical rating assigned to the item.

## Cold Start

Users without a learned positive genre profile use the train-only popularity
baseline.

This includes users absent from training and users whose training history
contains no relevant genre-bearing interactions.

## Determinism

Recommendation ordering uses deterministic scores and stable tie-breaking.

No random seed is introduced because the algorithm contains no stochastic
operation.

## Leakage Protection

Genre profiles and popularity statistics use only training interactions.

Movie genre metadata is static item-side information.

Validation outcomes are not used for model fitting or candidate scoring.

The locked test partition is not read.

## Evaluation

This checkpoint generates personalized Top-20 recommendations.

Ranking metrics are intentionally deferred to the dedicated evaluation
checkpoint, where popularity and genre-aware recommendations will be compared
using the same validation ground truth.