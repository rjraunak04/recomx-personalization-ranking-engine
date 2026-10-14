# RecomX Recommendation Diagnostics

## Purpose

Ranking accuracy alone does not describe the behaviour of a recommendation
system.

CP-07 evaluates popularity bias, catalogue exposure, long-tail behaviour and
performance across warm-user, cold-user and routing-strategy segments.

## Catalogue Coverage

Catalogue coverage measures the fraction of the complete movie catalogue that
appears at least once in recommendation output.

Higher coverage indicates broader catalogue exposure.

## Eligible-Item Coverage

The Day-02 models primarily operate on items with relevant training feedback.

Eligible-item coverage measures how much of this train-supported item universe
is exposed by the recommendation system.

## Recommendation Popularity

Average recommendation popularity is the mean number of relevant training
interactions associated with recommended items.

A very high value indicates concentration on globally popular content.

## Long-Tail Exposure

Items are ordered by relevant training interaction count.

The top 10 percent of popularity-ranked eligible items are defined as the
popularity head.

Items outside this head form the long tail.

Long-tail share measures the fraction of recommendation rows belonging to
this long-tail set.

## Warm Users

A warm validation user exists in the training partition.

The genre-aware model can use training history for personalization when that
history produces a valid positive genre profile.

## Cold Users

A cold validation user has no training history.

Cold users cannot receive history-based personalization and therefore require
a fallback strategy.

For the Day-02 hybrid baseline, the fallback is train-only global popularity.

## Strategy-Level Evaluation

The genre-aware recommendation output records the routing strategy used for
each user:

- `genre_personalized`
- `popularity_fallback`

Evaluating these groups separately prevents aggregate metrics from hiding the
behaviour of the personalized component.

## Cold-Item Exposure

A cold item is an item absent from the training interaction partition.

Cold-item exposure measures whether recommendation output contains such items.

The current Day-02 methods are expected to have limited or zero cold-item
capability because popularity and positive genre scoring are derived primarily
from training-supported items.

Later content or embedding systems can explicitly address item cold start.

## Leakage Protection

All popularity statistics are derived from training data.

Validation information is used only for offline evaluation and predefined
user cohorts.

The locked test partition remains untouched.

## Interpretation

A production recommendation system should not be selected using one metric
alone.

RecomX therefore considers:

- ranking accuracy
- warm-user performance
- cold-user behaviour
- catalogue coverage
- eligible-item coverage
- popularity concentration
- long-tail exposure
- cold-item exposure

These diagnostics establish the baseline behaviour that later collaborative
filtering, retrieval and ranking models must improve upon.