# Day-03 CP-02: Sparse Interaction Matrix

CP-02 converts train-only MovieLens feedback into a deterministic binary positive-feedback matrix.

Ratings >= 4.0 become positive interactions with value 1.0. Lower ratings are not encoded as positive evidence, and duplicate positive user-movie pairs are collapsed.

External userId and movieId values are sorted before contiguous matrix indices are assigned, making mappings independent of input row order.

The canonical representation is SciPy CSR with float32 values. A dense user-item matrix is intentionally forbidden at MovieLens 25M scale. The audit reports sparse storage and the theoretical dense float32 allocation avoided.

Only data/processed/ratings_train.parquet is read. Validation and locked-test ratings are not used. Generated artifacts live under reports/baselines/collaborative/sparse/ and remain ignored by Git.

CP-03 and CP-04 reuse this representation for Item-KNN and latent-factor learning. Cold validation users are absent from the collaborative matrix and later use the frozen popularity fallback.
