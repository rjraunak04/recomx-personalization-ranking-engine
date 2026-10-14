# RecomX Day-03: Collaborative Filtering and Latent-Factor Retrieval

## Objective

Day-03 adds scalable collaborative retrieval while preserving the frozen
Day-02 temporal evaluation protocol. Models train only on chronological
training data; the locked test partition remains untouched.

## System

Positive collaborative feedback is defined as rating >= 4.0. The training
signal is stored as a float32 CSR matrix with 137,682 positive-history users,
24,740 positive-history items and 9,981,093 positive user-item pairs.

Two collaborative approaches were implemented:

- sparse Item-KNN with cosine similarity and up to 100 neighbors per item
- 64-factor truncated SVD using SciPy sparse linear algebra

Dense user-item reconstruction is forbidden. Cold users use the train-only
popularity fallback and all train-seen items are filtered.

## Computational Result

The full Item-KNN graph contains 2,432,856 edges and required about 3 hours
7 minutes to build on the development machine. Its artifact is therefore
cached and reused.

The 64-factor SVD fit required about 13.34 seconds and stores approximately
41.6 MB of factor arrays. This establishes an important engineering trade-off:
Item-KNN is simple and interpretable but expensive to preprocess at this scale,
while sparse SVD is dramatically cheaper to fit.

## Accuracy

All models are evaluated on the same 17,847 users and 1,231,511 relevant
validation interactions.

At K=20:

| Model | Precision | Recall | NDCG | MAP | HitRate |
|---|---:|---:|---:|---:|---:|
| Item-KNN | 0.226363 | 0.093313 | 0.262566 | 0.145366 | 0.778450 |
| Latent SVD | 0.226125 | 0.093019 | 0.262225 | 0.145082 | 0.777273 |
| Popularity | 0.225399 | 0.092440 | 0.261514 | 0.144903 | 0.775032 |
| Genre-aware | 0.208668 | 0.079762 | 0.239643 | 0.135202 | 0.670757 |

Item-KNN is the strongest aggregate Day-03 model, although its gain over
popularity is small.

## Routing and Diagnostics

A complete collaborative Top-20 was produced for 592 Item-KNN users and 515
SVD users. The remaining users use popularity fallback. This explains why
aggregate metrics remain close to popularity and prevents overclaiming the
collaborative gain.

For the users actually served by the collaborative strategy at K=20:

| Strategy | Users | Precision | Recall | NDCG | MAP | HitRate |
|---|---:|---:|---:|---:|---:|---:|
| Item-KNN | 592 | 0.144932 | 0.101617 | 0.178606 | 0.088329 | 0.685811 |
| Latent SVD | 515 | 0.148932 | 0.107779 | 0.187029 | 0.091827 | 0.683495 |

SVD is stronger on most ranking-quality metrics inside its routed collaborative
subset, despite Item-KNN winning the aggregate benchmark.

Item-KNN exposes 804 unique items and SVD 725. Their catalogue coverage is
approximately 1.29% and 1.16%. Both expose zero cold-training items, and
long-tail exposure is effectively zero. Collaborative accuracy therefore
improved without solving catalogue concentration.

## Engineering Interpretation

Day-03 demonstrates why recommendation systems require more than a single
offline accuracy number. Aggregate results are affected heavily by fallback
traffic, while strategy-level evaluation reveals different behavior.

The next system stage should combine collaborative and content signals,
increase candidate breadth, preserve cold-start support and create a larger
candidate pool for a downstream ranker rather than treating Top-20 retrieval
as the final architecture.

## Guarantees

- chronological train/validation boundary preserved
- locked test untouched
- sparse collaborative representation
- deterministic mappings and ranking tie-breaks
- complete Top-20 output for every evaluable user
- all-train-seen filtering
- explicit cold/fallback routing
- same ranking metrics across every benchmark model
- warm/cold and strategy diagnostics
- generated artifacts excluded from Git
- expensive collaborative artifacts reused rather than rebuilt
