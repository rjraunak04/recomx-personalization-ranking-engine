# Day-06 — Recommendation Serving

## Objective
Turn RecomX from an offline recommendation experiment into a deterministic serving layer that can expose personalized Top-K recommendations through an API.

## CP-01: Serving contract
The contract is framework-independent so recommendation behavior can be tested without a running server.

Request: positive integer user_id and top_k from 1 to 20, default 20.

Response: user, serving strategy, and ordered unique recommendations containing movie_id, contiguous rank, numeric score, and reason.

Known users use personalized serving. Unseen users use cold_start_popularity. Personalized results filter seen training items. movieId is the deterministic final tie-break. Locked test access is forbidden during Day-06 development.
