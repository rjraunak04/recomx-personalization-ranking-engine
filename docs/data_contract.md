# H&M Dataset Contract

## Dataset

RecomX uses the H&M Personalized Fashion Recommendations dataset.

Source:

Kaggle competition: `h-and-m-personalized-fashion-recommendations`

Only the structured recommendation data is required for the initial
RecomX pipeline. Product images are not required during the data
foundation stage.

## Recommendation Problem

The project treats historical purchases as implicit user feedback.

A purchase indicates observed user-item interaction, but the absence of
a purchase must not automatically be interpreted as explicit dislike.

The core recommendation entities are:

- User: `customer_id`
- Item: `article_id`
- Interaction timestamp: `t_dat`
- Interaction: purchase
- Feedback: implicit

The long-term objective is personalized Top-K recommendation.

## Required Raw Files

### transactions_train.csv

Historical purchase interactions.

Required columns:

- t_dat
- customer_id
- article_id
- price
- sales_channel_id

This table forms the main user-item interaction dataset.

### articles.csv

Product catalogue and item metadata.

Important information includes:

- article_id
- product information
- product type
- colour
- department
- index
- section
- garment group
- textual description

The `article_id` field connects article metadata to transactions.

### customers.csv

Customer metadata.

Required fields include:

- customer_id
- FN
- Active
- club_member_status
- fashion_news_frequency
- age
- postal_code

The `customer_id` field connects customer metadata to transactions.

## Recommendation Relationships

The logical relationship is:

customers.csv
    |
    | customer_id
    v
transactions_train.csv
    ^
    | article_id
    |
articles.csv

The transaction table therefore acts as the interaction table between
users and items.

## Feedback Semantics

RecomX uses implicit feedback.

Observed purchase:

    user interacted with item

No observed purchase:

    unknown preference

Therefore, missing user-item interactions are not treated as explicit
negative ratings.

## Temporal Evaluation

Recommendation evaluation must respect interaction chronology.

Older interactions will be used to construct training history and later
interactions will be reserved for validation/test evaluation.

Random row splitting is not the default evaluation strategy because it
can allow future user behaviour to influence training.

The concrete temporal boundaries will be chosen only after the real
transaction date distribution is inspected.

## Data Integrity Expectations

The raw data layer must preserve the downloaded source files.

Raw CSV files must not be manually modified.

Later pipeline stages will validate:

- required columns
- identifier integrity
- timestamp parsing
- missing values
- duplicate interactions
- transaction ranges
- user/item relationships
- recommendation-specific statistics

## Git Policy

Large raw datasets are excluded from Git.

Only `.gitkeep` is tracked inside the raw data directory.

Generated interim and processed datasets are also excluded from source
control.

## CP-02 Boundary

This checkpoint defines the dataset and its contract.

It does not yet:

- transform data
- impute missing values
- engineer recommendation features
- construct negative samples
- calculate similarity
- train recommenders
- create embeddings
- perform ranking
- select temporal split dates

Those operations belong to later RecomX checkpoints.
