# RecomX Reproducibility Strategy

## Objective

CP-07 verifies that the RecomX data foundation produces deterministic
temporal partitions from the same validated interaction dataset.

Reproducibility is treated as a data-engineering and ML-system property,
not merely as setting a random seed.

## Deterministic Pipeline

The Day-01 data flow is:

MovieLens raw data

→ schema-aware ingestion

→ recommendation-specific validation

→ exploratory analysis

→ deterministic temporal splitting

→ reproducibility audit

The temporal split does not use random sampling.

Given identical input data and split configuration, partition membership
must remain identical.

## Logical Dataset Fingerprints

RecomX uses SHA-256 fingerprints over canonical interaction content.

Each partition is normalized using:

- userId
- movieId
- rating
- timestamp
- deterministic row ordering
- stable timestamp representation
- stable text serialization

The project intentionally does not use raw Parquet file hashes as the
primary logical-data fingerprint.

Parquet bytes may change because of compression settings, metadata or
library versions even when the logical interaction data is unchanged.

## Partition Audit

Independent fingerprints are produced for:

- train
- validation
- locked test

A combined SHA-256 fingerprint represents the complete temporal split.

The reproducibility verification runs the split twice from the same source
dataset and requires identical fingerprints.

## Locked Test Policy

The locked-test partition remains unavailable for model-selection decisions.

It must not influence:

- feature engineering decisions
- hyperparameter tuning
- retrieval strategy selection
- ranking model selection
- recommendation threshold selection

Its fingerprint exists for auditability, not for iterative model tuning.

## Generated Manifest

The audit manifest is written to:

`reports/reproducibility/manifest.json`

The manifest contains:

- algorithm identifier
- source row count
- partition row counts
- unique user counts
- unique item counts
- temporal ranges
- SHA-256 fingerprints
- combined split fingerprint

Generated audit artifacts are excluded from Git.

## Random Seeds

CP-07 does not introduce a fake random seed into deterministic operations.

Future stochastic model-training stages must explicitly define and record
their seeds.

## Reproducibility Principle

A production ML system should be able to answer:

"What exact data produced this experiment?"

The partition fingerprints and audit manifest establish the first layer of
that traceability for RecomX.
