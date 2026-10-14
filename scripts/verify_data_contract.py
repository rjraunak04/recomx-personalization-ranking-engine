"""Verify the static RecomX H&M dataset contract."""

from recomx.data.contracts import (
    ARTICLES_CONTRACT,
    CUSTOMERS_CONTRACT,
    DATASET_CONTRACTS,
    FEEDBACK_TYPE,
    INTERACTION_TYPE,
    ITEM_ID_COLUMN,
    RECOMMENDATION_TASK,
    TIMESTAMP_COLUMN,
    TRANSACTIONS_CONTRACT,
    USER_ID_COLUMN,
    expected_raw_files,
    missing_raw_files,
)


def main() -> None:
    """Run CP-02 dataset-contract checks."""

    assert len(DATASET_CONTRACTS) == 3

    assert TRANSACTIONS_CONTRACT.filename == "transactions_train.csv"
    assert ARTICLES_CONTRACT.filename == "articles.csv"
    assert CUSTOMERS_CONTRACT.filename == "customers.csv"

    assert USER_ID_COLUMN == "customer_id"
    assert ITEM_ID_COLUMN == "article_id"
    assert TIMESTAMP_COLUMN == "t_dat"

    assert USER_ID_COLUMN in TRANSACTIONS_CONTRACT.required_columns
    assert ITEM_ID_COLUMN in TRANSACTIONS_CONTRACT.required_columns
    assert TIMESTAMP_COLUMN in TRANSACTIONS_CONTRACT.required_columns

    assert USER_ID_COLUMN in CUSTOMERS_CONTRACT.required_columns
    assert ITEM_ID_COLUMN in ARTICLES_CONTRACT.required_columns

    assert FEEDBACK_TYPE == "implicit"
    assert INTERACTION_TYPE == "purchase"
    assert RECOMMENDATION_TASK == "personalized_top_k_recommendation"

    paths = expected_raw_files()

    assert set(paths) == {
        "transactions",
        "articles",
        "customers",
    }

    print("RecomX CP-02 data contract PASSED")
    print()
    print("Recommendation task:", RECOMMENDATION_TASK)
    print("Feedback type:", FEEDBACK_TYPE)
    print("Interaction:", INTERACTION_TYPE)
    print("User ID:", USER_ID_COLUMN)
    print("Item ID:", ITEM_ID_COLUMN)
    print("Timestamp:", TIMESTAMP_COLUMN)

    print()
    print("Expected raw files:")

    for name, path in paths.items():
        status = "FOUND" if path.is_file() else "NOT DOWNLOADED"
        print(f"  {name}: {path} [{status}]")

    missing = missing_raw_files()

    if missing:
        print()
        print(
            "Dataset files are not all present yet. "
            "This is acceptable for the static CP-02 contract check."
        )
    else:
        print()
        print("All required H&M raw files are available.")


if __name__ == "__main__":
    main()
