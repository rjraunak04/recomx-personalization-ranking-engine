# Day-03 CP-07: Collaborative Diagnostics

Aggregate ranking metrics can hide routing behavior because CP-05 sends cold
users, and warm users without a complete collaborative Top-20, to the
train-only popularity fallback. CP-07 therefore decomposes recommendation
quality and exposure.

The checkpoint reports warm versus cold ranking metrics, collaborative versus
fallback strategy metrics, catalogue and eligible-item coverage, unique item
exposure, average train popularity, long-tail share, and cold-item exposure.

All diagnostics consume frozen recommendation artifacts. Models are not
retrained, candidates are not regenerated, and the locked test partition is
not read.

The purpose is diagnostic rather than promotional: a collaborative model can
improve aggregate accuracy while still serving only a small warm-user cohort,
and a higher-accuracy model may concentrate exposure on popular items. These
trade-offs are retained for the final Day-03 benchmark.
