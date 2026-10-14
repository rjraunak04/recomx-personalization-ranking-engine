# RecomX Day-04 Results — Hybrid Candidate Retrieval

Day-04 introduces a deterministic multi-source retrieval layer with a target
budget of 100 unique candidates per user. It combines Item-KNN, latent SVD,
genre/content and popularity candidates through reciprocal-rank fusion.

The stage deliberately separates retrieval from final ranking. Its primary
goal is to increase the probability that relevant items enter the candidate
set while retaining multiple retrieval signals for a later learning-to-rank
model.

## Engineering guarantees

- chronological training boundary preserved
- locked test never read
- existing expensive Day-03 artifacts reused
- all train-seen items filtered again at fusion time
- user-item candidates deduplicated across sources
- deterministic score and movie-id tie breaking
- source provenance retained per candidate
- candidate budget capped at 100
- candidate Recall@20/50/100 reported
- source contribution audited
- generated artifacts remain outside Git

The benchmark intentionally records the achieved candidate depth. Day-03
artifacts contain Top-20 lists, so a union of four sources can produce fewer
than 100 unique candidates for some users. This is reported rather than
silently padding candidates.
