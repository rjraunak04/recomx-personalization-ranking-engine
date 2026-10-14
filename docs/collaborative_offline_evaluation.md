# Day-03 CP-06: Collaborative Offline Evaluation

CP-06 performs the first fair accuracy comparison between the Day-02 baselines
and Day-03 collaborative recommenders.

## Fair comparison

All four models use the exact same frozen validation ground truth and the same
macro-averaged ranking implementation:

- Precision@5/10/20
- Recall@5/10/20
- NDCG@5/10/20
- MAP@5/10/20
- HitRate@5/10/20

The models are popularity, genre-personalized, Item-KNN, and latent SVD.

No model is retrained and no candidate artifact is regenerated in this
checkpoint. The locked test partition remains untouched.

## Interpretation caution

CP-05 deliberately routes cold users to popularity and also falls back when a
collaborative model cannot produce a complete Top-20 after the all-train-seen
filter. Therefore aggregate metrics can remain close to popularity. CP-07 will
separate warm/cold and strategy cohorts so that collaborative behavior is not
hidden by fallback traffic.
