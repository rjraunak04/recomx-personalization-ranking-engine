"""Run RecomX CP-04 recommendation data-quality validation."""

from recomx.data.validate import (
    assert_required_rules,
    run_data_quality_validation,
)


def main() -> None:
    print("RecomX CP-04 Data Quality Validation")
    print("=" * 72)

    results = run_data_quality_validation()

    for result in results:
        status = "PASS" if result.passed else "FAIL"

        print(f"{status:<5} | {result.rule:<36} | value={result.value}")

    passed = sum(result.passed for result in results)

    total = len(results)

    print()
    print(f"Rules passed: {passed}/{total}")

    observed_blank_tags = next(
        (result.value for result in results if result.rule == "tags_nonempty_observation"),
        0,
    )

    print(
        "Observed blank auxiliary tags:",
        observed_blank_tags,
    )

    assert_required_rules(results)

    print()
    print("CP-04 PASSED")


if __name__ == "__main__":
    main()
