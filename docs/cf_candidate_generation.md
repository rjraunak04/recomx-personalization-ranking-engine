# Day-03 CP-05: Collaborative Candidate Generation

CP-05 converts the frozen Item-KNN and latent-SVD artifacts into actual Top-20
recommendations for the same evaluable validation-user cohort established on
Day-02.

## Routing

Warm users are scored by the collaborative model. Cold users receive the
frozen train-only popularity fallback. If a collaborative model cannot produce
a complete Top-20 list, the user falls back to popularity rather than emitting
an incomplete evaluation row set.

## Seen-item safety

The CP-02 collaborative matrix contains positive feedback only, while the
recommendation contract requires filtering **all** train-seen items. CP-05
therefore applies a second filter using the complete training interaction
history, including ratings below the relevance threshold.

This distinction prevents already-consumed but non-positive items from leaking
into recommendations.

## Artifact reuse

The expensive CP-03 Item-KNN graph and CP-04 SVD factors are loaded from disk;
they are not retrained. Outputs are generated under the ignored reports tree.

Validation ratings and the locked test partition are never read by this
checkpoint. The validation cohort file supplies only the previously frozen
evaluable user IDs.
