# RecomX Recommendation-Specific EDA

## Objective

CP-05 studies the structure of MovieLens 25M from a recommendation-system
perspective rather than performing generic tabular profiling.

The analysis focuses on signals that directly affect retrieval, ranking,
evaluation and cold-start strategy.

## Interaction Matrix

The user-item interaction matrix is characterized using:

- unique users
- unique rated items
- observed user-item pairs
- theoretical user-item pairs
- interaction density
- sparsity

High sparsity is expected in real recommendation systems because each user
interacts with only a small fraction of the available catalogue.

## User Activity

Interactions per user are measured using:

- mean
- median
- 95th percentile
- full activity distribution

This identifies differences between light and highly active users.

## Item Popularity

Interactions per item are analyzed using:

- mean
- median
- 95th percentile
- popularity distribution
- ranked popularity curve
- top 1 percent interaction share
- top 10 percent interaction share

These measurements expose popularity concentration and long-tail behaviour.

## Rating Behaviour

The explicit-feedback distribution is measured across the MovieLens rating
scale.

No positive-preference threshold is introduced during EDA. Transforming
explicit ratings into implicit or ranking labels remains a later modeling
decision.

## Temporal Behaviour

Rating interactions are aggregated by month to understand how interaction
volume changes over time.

The timestamp remains essential for future leakage-safe temporal splitting.

## Catalogue Coverage

The analysis measures the fraction of catalogue items that appear in the
rating interaction table.

Unrated catalogue items are important for later cold-start and content-based
recommendation decisions.

## Generated Artifacts

Machine-readable EDA outputs are stored under:

`reports/eda/`

Visual outputs are stored under:

`reports/figures/`

Generated reports and figures are analytical artifacts and should not be
treated as source data.

## Modeling Implications

CP-05 results will inform:

- temporal train/validation/test design
- candidate generation
- popularity baselines
- cold-start handling
- ranking evaluation
- long-tail analysis

No recommendation model is trained during CP-05.
