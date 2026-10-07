"""Dataset contracts for the H&M recommendation dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetFileContract:
    """Contract describing one required raw dataset file."""

    name: str
    filename: str
    description: str
    required_columns: tuple[str, ...]


TRANSACTIONS_CONTRACT = DatasetFileContract(
    name="transactions",
    filename="transactions_train.csv",
    description="Historical customer purchase interactions.",
    required_columns=(
        "t_dat",
        "customer_id",
        "article_id",
        "price",
        "sales_channel_id",
    ),
)

ARTICLES_CONTRACT = DatasetFileContract(
    name="articles",
    filename="articles.csv",
    description="Item catalogue and product metadata.",
    required_columns=(
        "article_id",
        "product_code",
        "prod_name",
        "product_type_no",
        "product_type_name",
        "product_group_name",
        "graphical_appearance_no",
        "graphical_appearance_name",
        "colour_group_code",
        "colour_group_name",
        "department_no",
        "department_name",
        "index_code",
        "index_name",
        "index_group_no",
        "index_group_name",
        "section_no",
        "section_name",
        "garment_group_no",
        "garment_group_name",
        "detail_desc",
    ),
)

CUSTOMERS_CONTRACT = DatasetFileContract(
    name="customers",
    filename="customers.csv",
    description="Customer metadata used for personalization and cold-start analysis.",
    required_columns=(
        "customer_id",
        "FN",
        "Active",
        "club_member_status",
        "fashion_news_frequency",
        "age",
        "postal_code",
    ),
)


DATASET_CONTRACTS = (
    TRANSACTIONS_CONTRACT,
    ARTICLES_CONTRACT,
    CUSTOMERS_CONTRACT,
)


USER_ID_COLUMN = "customer_id"
ITEM_ID_COLUMN = "article_id"
TIMESTAMP_COLUMN = "t_dat"

INTERACTION_TYPE = "purchase"
FEEDBACK_TYPE = "implicit"
RECOMMENDATION_TASK = "personalized_top_k_recommendation"


def expected_raw_files(raw_dir: str | Path = "data/raw") -> dict[str, Path]:
    """Return expected raw dataset paths without reading the files."""

    root = Path(raw_dir)

    return {
        contract.name: root / contract.filename
        for contract in DATASET_CONTRACTS
    }


def missing_raw_files(raw_dir: str | Path = "data/raw") -> list[Path]:
    """Return raw dataset files that are not currently available."""

    return [
        path
        for path in expected_raw_files(raw_dir).values()
        if not path.is_file()
    ]
