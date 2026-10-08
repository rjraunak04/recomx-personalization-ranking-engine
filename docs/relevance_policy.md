# RecomX Explicit-Feedback Relevance Policy

## Purpose

MovieLens provides explicit ratings rather than binary positive interactions.

A recommendation evaluation pipeline must therefore define which future
interactions represent relevant outcomes.

## Primary Day-02 Policy

The initial RecomX relevance rule is:

`rating >= 4.0`

Ratings of 4.0, 4.5 and 5.0 are treated as relevant outcomes.

Ratings below 4.0 are not treated as positive Top-K targets.

## Why All Ratings Are Not Positive

An interaction only proves that a user rated an item.

It does not imply that the user liked the item.

Treating a rating such as 1.0 or 2.0 as a positive recommendation target would
reward a recommender for recovering items associated with negative explicit
feedback.

## Configuration

The relevance threshold is part of the central evaluation protocol and can be
changed deliberately for later sensitivity analysis.

The threshold must not be silently duplicated across models.

## Leakage Rule

Relevance profiling during model development is performed on the training
partition.

Validation relevance is used later to construct evaluation ground truth.

Locked-test outcomes remain isolated from Day-02 model development.

## Interpretation

The relevance threshold transforms explicit feedback into a binary evaluation
concept:

- relevant: rating >= 4.0
- not relevant: rating < 4.0

This transformation is used for recommendation evaluation. It does not erase
or overwrite the original explicit rating values.

## Reproducibility

The profiling script records:

- total training interactions
- relevant interactions
- non-relevant interactions
- relevance rate
- total and relevant users
- total and relevant items

Generated profiling artifacts are analytical outputs and are excluded from
version control.