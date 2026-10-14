# RecomX Day-01 Data Architecture

## Purpose

Day 01 establishes the trustworthy data layer required by all later RecomX recommendation components.

No recommendation algorithm is allowed to bypass this foundation.

## Data Flow

```text
MovieLens 25M CSV Sources
          |
          v
+-------------------------+
| Schema-Aware Loading    |
| load.py                 |
+-------------------------+
          |
          v
+-------------------------+
| Parquet Ingestion       |
| ingest.py               |
+-------------------------+
          |
          v
+-------------------------+
| Data Quality Validation |
| validate.py             |
+-------------------------+
          |
          v
+-------------------------+
| Recommendation EDA      |
| eda.py                  |
+-------------------------+
          |
          v
+-------------------------+
| Temporal Splitting      |
| split.py                |
+-------------------------+
          |
          v
+-------------------------+
| Reproducibility Audit   |
| reproducibility.py      |
+-------------------------+
          |
          v
Train / Validation / Locked Test