# RecomX Leakage-Safe Temporal Splitting

## Objective

CP-06 creates chronological train, validation and locked-test partitions
for recommendation-system development.

Random splitting is intentionally avoided because recommendation data
contains temporal interaction history.

A random split can allow future user behaviour to influence training data
used to predict earlier behaviour.

## Split Strategy

The interaction stream is ordered chronologically and divided approximately
into:

- 80 percent training
- 10 percent validation
- 10 percent locked test

The exact fractions may differ slightly because all interactions sharing the
same timestamp remain within one partition.

## Partition Semantics

Training contains the earliest interaction history.

Validation contains later interactions and may be used for model selection,
hyperparameter tuning and recommendation-policy development.

Locked test contains the latest interactions.

The locked test partition must not be used for:

- feature selection
- hyperparameter tuning
- candidate-generation tuning
- ranking-model selection
- threshold selection

It is reserved for final unbiased evaluation.

## Timestamp Boundary Safety

RecomX does not split identical timestamps across partitions.

The required temporal relationship is:

train max timestamp < validation min timestamp

validation max timestamp < locked-test min timestamp

This prevents boundary timestamp leakage.

## Cold Start

Temporal splitting naturally exposes cold-start entities.

CP-06 records:

- validation users unseen in training
- validation items unseen in training
- test users unseen in training
- test items unseen in training
- test users unseen in all pre-test history
- test items unseen in all pre-test history

These statistics will later guide fallback and hybrid recommendation
strategies.

## Generated Data

The following generated partitions are written under `data/processed/`:

- `ratings_train.parquet`
- `ratings_validation.parquet`
- `ratings_locked_test.parquet`

These are generated data artifacts and are excluded from Git.

## Audit Artifact

Split metadata is written to:

`reports/splits/temporal_split_summary.json`

The audit report records:

- row counts
- actual split fractions
- temporal boundaries
- cold-start counts

Generated split reports are excluded from source control.

## Reproducibility

The splitting procedure is deterministic.

Given the same source dataset and configuration, the same chronological
partitions and cutoff timestamps are produced.

No random seed is required because no random sampling is performed.
