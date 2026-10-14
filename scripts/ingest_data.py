"""Run MovieLens 25M ingestion."""

from recomx.data.ingest import ingest_movielens


def main() -> None:
    print("Starting RecomX MovieLens 25M ingestion...")
    ingest_movielens()
    print()
    print("CP-03 ingestion completed.")


if __name__ == "__main__":
    main()
