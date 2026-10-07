# RecomX Data Quality Validation

## Purpose

CP-04 determines whether the MovieLens 25M interim dataset is
structurally and logically suitable for recommendation-system
development.

Successful ingestion does not automatically imply modeling readiness.

## Ratings Integrity

The primary interaction table is validated for:

- required columns
- missing required values
- positive user identifiers
- positive movie identifiers
- valid 0.5 to 5.0 rating domain
- half-star rating increments
- exact duplicate events
- repeated user-movie interactions
- valid timestamps

Repeated user-movie interactions are measured instead of automatically
removed because interaction-history handling is a downstream temporal
modeling decision.

## Movie Catalogue Integrity

The movie catalogue is validated for:

- required columns
- missing movie identifiers
- positive movie identifiers
- duplicate movie identifiers
- blank titles
- missing genre fields

## Tag Integrity

User-generated tag interactions are validated for:

- required columns
- user identifiers
- movie identifiers
- timestamps
- blank tag text

Blank tag text is treated as an observed non-blocking auxiliary-content
quality issue.

The raw and interim datasets remain immutable.

Blank tags will be excluded when model-ready content features are
constructed downstream.

## Genome Integrity

Genome data is validated for:

- required columns
- missing values
- relevance values between 0 and 1
- unique movie-tag pairs
- unique genome tag IDs
- non-empty genome vocabulary
- valid genome tag foreign keys

## Cross-Table Integrity

RecomX validates relationships between:

- ratings and movies
- tags and movies
- genome scores and movies
- genome scores and genome tag vocabulary

Catalogue rating coverage is also measured.

## Observed MovieLens Finding

CP-04 detected 25 blank user-generated tag values.

These rows do not invalidate the primary recommendation interaction
dataset because tags are auxiliary content signals rather than required
rating interactions.

The issue remains explicitly measured instead of being silently hidden.

## Modeling Boundary

CP-04 does not define which rating represents positive preference.

MovieLens contains explicit feedback. Converting ratings into positive
or negative ranking interactions is a later modeling decision.

## Next Stage

CP-05 performs recommendation-specific exploratory analysis, including:

- interaction density
- matrix sparsity
- user activity
- item popularity
- long-tail behaviour
- rating distribution
- temporal interaction patterns
