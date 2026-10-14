# RecomX Day-03: Collaborative Filtering and Latent-Factor Retrieval

## Objective

Day-03 introduces collaborative recommendation signals while preserving the
leakage-safe Day-02 evaluation contract. The goal is to improve warm-user
personalization without hiding cold-start limitations or sacrificing
reproducibility.

## Frozen Experiment Contract

- fit models only on the chronological training partition
- use validation only for development and model selection
- do not read the locked test partition
- define positive feedback as MovieLens rating >= 4.0
- encode positive interactions as binary value 1.0
- filter every item previously seen by a user in training
- evaluate at K = 5, 10 and 20
- reuse the Day-02 validation ground truth and ranking metrics
- route users without training history to the train-only popularity fallback

## Scale-Safe Representation

MovieLens 25M is not represented as a dense user-item array. Day-03 uses a
SciPy-compatible CSR sparse matrix with float32 values and deterministic
user/item mappings. Materializing a dense matrix is explicitly forbidden by
the experiment contract.

This decision keeps memory proportional to observed positive interactions
rather than all possible user-item pairs.

## Collaborative Baselines

Two complementary collaborative approaches are planned.

### Item-KNN

Item-KNN uses cosine neighborhood similarity over sparse interaction vectors.
A user's positive history contributes neighboring items as candidates. The
implementation must retain only bounded neighborhoods and must never
materialize a dense all-item similarity matrix.

### Latent-Factor Retrieval

The latent baseline uses deterministic truncated SVD with 64 factors and
random seed 42. User and item representations are used for batched candidate
scoring so evaluation does not require an all-users by all-items score matrix
in memory.

## Candidate and Fallback Policy

Warm users receive collaborative candidates with all training-seen items
removed. Cold users have no collaborative representation, so they are routed
to the frozen train-only popularity baseline rather than receiving fabricated
personalization.

Every model is expected to return a deterministic Top-20 list for each
evaluable validation user.

## Evaluation

Day-03 reuses the Day-02 validation ground truth and the same metrics:

- Precision@K
- Recall@K
- NDCG@K
- MAP@K
- HitRate@K

Later checkpoints also compare catalogue coverage, eligible-item coverage,
recommendation popularity, long-tail exposure and cold-item exposure.

The benchmark will compare collaborative models against both Day-02 baselines.
A collaborative model is not tuned merely to force it to beat popularity; a
negative result remains a valid experimental finding.

## Checkpoints

1. collaborative-filtering experiment contract
2. sparse interaction matrix and deterministic mappings
3. scalable Item-KNN recommender
4. latent-factor recommender
5. candidate generation, seen filtering and cold fallback
6. offline ranking evaluation against Day-02 baselines
7. cohort and beyond-accuracy diagnostics
8. reproducible Day-03 benchmark and project freeze

## Leakage Boundary

The locked test partition is intentionally outside Day-03 development.
Hyperparameters, thresholds and model choices must not use it. Final test
evaluation belongs to a later explicitly frozen project stage.
