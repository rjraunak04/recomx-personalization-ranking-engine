# Day-03 CP-04: Latent-Factor SVD Retrieval

CP-04 learns compact collaborative user and item representations from the
frozen CP-02 train-only binary positive-feedback CSR matrix.

## Model

The experiment contract fixes 64 latent factors and random state 42. RecomX
uses `scipy.sparse.linalg.svds`, so the 137k by 24k user-item matrix remains
sparse during factorization.

The singular values are split symmetrically between user and item factors:

`user_embedding = U * sqrt(S)`

`item_embedding = V * sqrt(S)`

Their dot product approximates the original positive-feedback interaction
matrix.

## Retrieval

Warm-user scores are computed from the compact user embedding against item
embeddings in bounded batches. Train-seen positive items are filtered before
Top-K selection. Equal scores use ascending external `movieId` as the
deterministic tie-break.

Cold users deliberately return no latent candidates at this layer; the frozen
Day-03 contract applies the train-only popularity fallback later.

## Leakage boundary

Only CP-02 train artifacts are loaded. Validation and locked-test interactions
are not read during fitting. CP-04 creates model artifacts only; comparative
offline validation remains a later checkpoint.
