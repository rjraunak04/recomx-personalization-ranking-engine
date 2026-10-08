# RecomX Day-02: Recommendation Baselines and Offline Evaluation

## Objective

Day-02 establishes a reproducible, leakage-safe recommendation benchmark on
MovieLens 25M.

The purpose is not to force a personalized model to outperform a simpler
baseline. The purpose is to create a trustworthy benchmark that later
collaborative-filtering, retrieval and ranking models must improve upon.

## Evaluation Protocol

The chronological Day-01 split is preserved.

- training partition: model fitting
- validation partition: development evaluation
- locked test partition: untouched
- explicit relevance threshold: rating >= 4.0
- ranking cutoffs: 5, 10 and 20

Relevant validation interactions form the evaluation ground truth.

Previously observed training items are excluded where required by the
evaluation protocol.

## Evaluation Population

The final validation benchmark contains:

- 17,847 evaluable users
- 1,231,511 relevant validation interactions

The validation population contains both warm users with training history and
cold users without training history.

## Popularity Baseline

The popularity baseline ranks movies by the number of relevant training
interactions.

Previously seen items are filtered for warm users.

All 17,847 evaluation users receive a complete Top-20 list.

The popularity baseline generated:

- 356,940 recommendation rows
- zero seen-item violations

## Genre-Aware Baseline

The personalized baseline learns normalized genre preferences from relevant
training history.

Its score combines:

`0.95 * genre affinity + 0.05 * normalized popularity`

Users without a usable genre profile fall back to train-only popularity.

Observed routing:

- 5,138 users: genre-personalized
- 12,709 users: popularity fallback

All users receive complete Top-20 recommendation lists with zero seen-item
violations.

## Ranking Results

Popularity was the stronger ranking-accuracy baseline at every evaluated
cutoff.

At K=10:

| Model | Precision | Recall | NDCG | MAP | HitRate |
|---|---:|---:|---:|---:|---:|
| Popularity | 0.268079 | 0.058249 | 0.294354 | 0.189196 | 0.719897 |
| Genre-aware | 0.248445 | 0.050453 | 0.271206 | 0.176884 | 0.631647 |

This result is retained rather than tuning the experiment merely to make the
personalized baseline appear stronger.

## Warm-User Finding

The largest weakness of the genre model appears among warm users.

At K=20:

| Model | Precision | Recall | NDCG | MAP | HitRate |
|---|---:|---:|---:|---:|---:|
| Popularity | 0.062925 | 0.047217 | 0.081204 | 0.034942 | 0.439503 |
| Genre-aware | 0.004933 | 0.003274 | 0.005396 | 0.001319 | 0.078073 |

The simple 19-genre representation is therefore insufficient as the main
personalization signal for accurate ranking.

This motivates stronger collaborative and hybrid representations in later
project stages.

## Cold-User Behaviour

Cold validation users have no training history.

Both systems therefore route these users through the popularity fallback.

Their ranking metrics are consequently identical.

This confirms deterministic fallback behaviour rather than pretending that
history-based personalization exists for unseen users.

## Beyond-Accuracy Trade-Off

The genre-aware system exposes a broader and less popularity-concentrated
item set.

Observed results:

| Diagnostic | Popularity | Genre-aware |
|---|---:|---:|
| Unique recommended items | 414 | 979 |
| Catalogue coverage | 0.006632 | 0.015683 |
| Eligible-item coverage | 0.016734 | 0.030841 |
| Average recommendation popularity | 36232.76 | 28067.76 |
| Long-tail share | 0.000000 | 0.148949 |
| Cold-item exposure share | 0.000000 | 0.047367 |

The genre model therefore increases catalogue breadth, long-tail exposure and
item cold-start exposure, but currently sacrifices substantial ranking
accuracy.

## Cold-Item Interpretation

Genre metadata allows the content-based model to expose some movies absent
from the training interaction history.

The popularity baseline cannot do this because its candidate universe is
derived directly from training popularity statistics.

This demonstrates the potential value of content information for item cold
start while also showing that content availability alone does not guarantee
good ranking quality.

## Engineering Guarantees

Day-02 verifies:

- train-only model fitting
- explicit relevance policy
- chronological validation
- locked-test isolation
- seen-item filtering
- deterministic recommendation ranking
- complete Top-20 output
- duplicate recommendation protection
- hand-tested ranking metrics
- warm/cold segmentation
- strategy-level evaluation
- beyond-accuracy diagnostics
- generated artifact isolation from Git

## Day-02 Conclusion

The strongest Day-02 accuracy baseline is global popularity.

The genre-aware baseline is useful as an interpretable content baseline and
improves recommendation breadth, but its warm-user ranking performance is too
weak to replace popularity.

Later RecomX stages should therefore aim to:

1. learn collaborative user-item structure,
2. improve personalized candidate retrieval,
3. retain robust cold-start fallback,
4. combine collaborative and content signals,
5. improve accuracy without collapsing catalogue diversity.

Day-02 therefore establishes an honest benchmark rather than an artificially
optimized result.