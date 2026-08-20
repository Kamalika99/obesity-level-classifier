"""
data_validation.py

Validation checks for the Obesity Levels dataset.
Reports issues rather than silently fixing them; cleaning decisions are
made explicitly in preprocessing.py with reasoning documented.
"""

import pandas as pd


EXPECTED_CATEGORICAL_VALUES = {
    "Gender": {"Female", "Male"},
    "family_history_with_overweight": {"yes", "no"},
    "FAVC": {"yes", "no"},
    "CAEC": {"no", "Sometimes", "Frequently", "Always"},
    "SMOKE": {"yes", "no"},
    "SCC": {"yes", "no"},
    "CALC": {"no", "Sometimes", "Frequently", "Always"},
    "MTRANS": {
        "Public_Transportation",
        "Walking",
        "Automobile",
        "Motorbike",
        "Bike",
    },
    "NObeyesdad": {
        "Insufficient_Weight",
        "Normal_Weight",
        "Overweight_Level_I",
        "Overweight_Level_II",
        "Obesity_Type_I",
        "Obesity_Type_II",
        "Obesity_Type_III",
    },
}

# Plausible physical ranges used only to flag suspicious rows, not to
# silently drop them.
NUMERICAL_RANGES = {
    "Age": (10, 100),
    "Height": (1.0, 2.3),   # meters
    "Weight": (20, 300),    # kg
    "FCVC": (1, 3),
    "NCP": (1, 4),
    "CH2O": (1, 3),
    "FAF": (0, 3),
    "TUE": (0, 2),
}


def validate_dataset(df: pd.DataFrame) -> dict:
    """
    Run a full set of validation checks on the raw obesity dataset.

    Returns a dictionary summarizing all findings. Does not modify df.
    """
    report = {}

    # 1. Missing values
    missing = df.isnull().sum()
    report["missing_values"] = missing[missing > 0].to_dict()

    # 2. Duplicate rows
    dup_mask = df.duplicated()
    report["duplicate_rows"] = int(dup_mask.sum())
    report["duplicate_indices"] = df[dup_mask].index.tolist()

    # 3. Data types
    report["dtypes"] = df.dtypes.astype(str).to_dict()

    # 4. Unexpected categorical values
    unexpected_categories = {}
    for col, expected in EXPECTED_CATEGORICAL_VALUES.items():
        if col in df.columns:
            actual = set(df[col].dropna().unique())
            unexpected = actual - expected
            if unexpected:
                unexpected_categories[col] = list(unexpected)
    report["unexpected_categorical_values"] = unexpected_categories

    # 5. Numerical out-of-range values
    out_of_range = {}
    for col, (lo, hi) in NUMERICAL_RANGES.items():
        if col in df.columns:
            bad = df[(df[col] < lo) | (df[col] > hi)]
            if len(bad) > 0:
                out_of_range[col] = len(bad)
    report["numerical_out_of_range_counts"] = out_of_range

    # 6. Logical consistency: Weight/Height should not be non-positive
    invalid_physical = df[(df["Height"] <= 0) | (df["Weight"] <= 0)]
    report["non_positive_height_or_weight_rows"] = len(invalid_physical)

    # 7. Target-label consistency: verify NObeyesdad only has the 7 expected labels
    if "NObeyesdad" in df.columns:
        target_values = set(df["NObeyesdad"].unique())
        report["target_classes_found"] = sorted(target_values)
        report["target_class_count"] = len(target_values)
        report["target_distribution"] = df["NObeyesdad"].value_counts().to_dict()

    return report


def print_validation_report(report: dict) -> None:
    """Pretty-print the validation report to stdout."""
    print("=" * 60)
    print("DATA VALIDATION REPORT")
    print("=" * 60)

    print("\nMissing values:")
    print(report["missing_values"] if report["missing_values"] else "  None found.")

    print(f"\nDuplicate rows: {report['duplicate_rows']}")

    print("\nUnexpected categorical values:")
    print(
        report["unexpected_categorical_values"]
        if report["unexpected_categorical_values"]
        else "  None found. All categories match expected values."
    )

    print("\nNumerical values out of plausible range:")
    print(
        report["numerical_out_of_range_counts"]
        if report["numerical_out_of_range_counts"]
        else "  None found."
    )

    print(f"\nNon-positive Height/Weight rows: {report['non_positive_height_or_weight_rows']}")

    print(f"\nTarget classes found ({report['target_class_count']}):")
    for cls in report["target_classes_found"]:
        print(f"  - {cls}: {report['target_distribution'][cls]} records")

    print("=" * 60)


if __name__ == "__main__":
    df = pd.read_csv("data/obesity_data.csv")
    report = validate_dataset(df)
    print_validation_report(report)
