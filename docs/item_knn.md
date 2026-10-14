# Day-03 CP-03: Sparse Item-KNN

CP-03 adds item-based collaborative filtering on top of the frozen CP-02
positive-feedback CSR matrix.

## Similarity

Each movie is represented by its sparse binary positive-user vector. Cosine
similarity measures overlap while normalizing for item interaction volume.

The implementation processes one sparse item row at a time. It never
materializes the full 24,740 by 24,740 item-similarity matrix.

## Bounded neighborhood graph

Only the strongest 100 positive-similarity neighbors are retained per item.
The resulting graph is CSR float32 and therefore bounded to at most
`number_of_items * 100` stored edges.

Self-neighbors are removed. Equal similarity scores are resolved
deterministically by ascending external `movieId`.

## Recommendation

For a warm user, similarity scores from all positively rated training-history
items are aggregated. Every positive train-history item is then removed from
the candidate set before deterministic Top-K ranking.

An unknown/cold user deliberately returns no collaborative candidates at this
layer. The frozen Day-03 contract routes that user to the train-only popularity
fallback in the later candidate-generation checkpoint.

## Leakage boundary

The Item-KNN graph is learned exclusively from the CP-02 train-only sparse
matrix. Validation and locked-test interactions are not read during fitting.

CP-03 builds the neighborhood retrieval layer only. Full validation-user
candidate generation and ranking evaluation are intentionally handled in later
Day-03 checkpoints.
