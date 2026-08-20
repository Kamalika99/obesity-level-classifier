"""
preprocessing.py

Reusable, leakage-safe preprocessing pipelines for the Obesity Levels
Classifier project.

Defines:
- Two feature configurations (full vs lifestyle-focused)
- A ColumnTransformer-based preprocessing pipeline for each configuration
- A helper to load, clean, and split the dataset

All encoding/scaling is fit only inside sklearn Pipelines on training
folds, never on the full dataset ahead of time, to avoid data leakage.
"""

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Feature groupings (derived from actual dataset inspection, not assumed)
# ---------------------------------------------------------------------------

TARGET_COLUMN = "NObeyesdad"

# Numerical features (continuous / count-like, from real inspection of dtypes)
NUMERICAL_FEATURES = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]

# Height/Weight are the body-measurement features under investigation
BODY_MEASUREMENT_FEATURES = ["Height", "Weight"]

# Nominal categorical (no inherent order)
NOMINAL_FEATURES = [
    "Gender",
    "family_history_with_overweight",
    "FAVC",
    "SMOKE",
    "SCC",
    "MTRANS",
]

# Ordinal categorical (clear increasing order in the underlying survey)
ORDINAL_FEATURES = ["CAEC", "CALC"]
ORDINAL_CATEGORY_ORDER = [
    ["no", "Sometimes", "Frequently", "Always"],  # CAEC
    ["no", "Sometimes", "Frequently", "Always"],  # CALC
]

ALL_FEATURES_FULL = NUMERICAL_FEATURES + NOMINAL_FEATURES + ORDINAL_FEATURES

# Lifestyle-focused: same as full, minus body measurements
NUMERICAL_FEATURES_LIFESTYLE = [
    f for f in NUMERICAL_FEATURES if f not in BODY_MEASUREMENT_FEATURES
]
ALL_FEATURES_LIFESTYLE = (
    NUMERICAL_FEATURES_LIFESTYLE + NOMINAL_FEATURES + ORDINAL_FEATURES
)


@dataclass
class FeatureConfig:
    name: str
    numerical: list
    nominal: list
    ordinal: list

    @property
    def all_features(self):
        return self.numerical + self.nominal + self.ordinal


FULL_CONFIG = FeatureConfig(
    name="full",
    numerical=NUMERICAL_FEATURES,
    nominal=NOMINAL_FEATURES,
    ordinal=ORDINAL_FEATURES,
)

LIFESTYLE_CONFIG = FeatureConfig(
    name="lifestyle",
    numerical=NUMERICAL_FEATURES_LIFESTYLE,
    nominal=NOMINAL_FEATURES,
    ordinal=ORDINAL_FEATURES,
)


# ---------------------------------------------------------------------------
# Loading & cleaning
# ---------------------------------------------------------------------------

def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    """
    Load the raw obesity dataset and apply documented, non-silent cleaning.

    Cleaning decisions:
    - Drop exact duplicate rows (24 found on inspection). Exact duplicates
      carry no new information and would otherwise bias class frequencies
      and inflate apparent sample size.
    """
    df = pd.read_csv(csv_path)
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_after = len(df)
    print(f"[load_and_clean_data] Dropped {n_before - n_after} exact duplicate rows "
          f"({n_before} -> {n_after}).")
    return df


def compute_bmi(df: pd.DataFrame) -> pd.Series:
    """Compute BMI = weight(kg) / height(m)^2, for EXPLORATORY analysis only."""
    return df["Weight"] / (df["Height"] ** 2)


# ---------------------------------------------------------------------------
# Preprocessing pipeline construction
# ---------------------------------------------------------------------------

def build_preprocessor(config: FeatureConfig) -> ColumnTransformer:
    """
    Build a ColumnTransformer for the given feature configuration.

    - Numerical features -> StandardScaler
    - Nominal categorical -> OneHotEncoder (handle_unknown='ignore')
    - Ordinal categorical -> OrdinalEncoder with explicit category order

    This transformer is meant to be embedded inside a sklearn Pipeline
    together with an estimator, so that fitting only ever happens on
    training folds/data (no leakage into validation/test).
    """
    transformers = []

    if config.numerical:
        transformers.append(("num", StandardScaler(), config.numerical))

    if config.nominal:
        transformers.append(
            ("nom", OneHotEncoder(handle_unknown="ignore", sparse_output=False), config.nominal)
        )

    if config.ordinal:
        transformers.append(
            (
                "ord",
                OrdinalEncoder(categories=ORDINAL_CATEGORY_ORDER),
                config.ordinal,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def get_train_test_split(df: pd.DataFrame, config: FeatureConfig, test_size: float = 0.2):
    """
    Perform a reproducible, stratified train/test split for a given
    feature configuration.

    Returns X_train, X_test, y_train, y_test (y as raw string labels).
    """
    X = df[config.all_features].copy()
    y = df[TARGET_COLUMN].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df = load_and_clean_data("data/obesity_data.csv")

    for config in [FULL_CONFIG, LIFESTYLE_CONFIG]:
        X_train, X_test, y_train, y_test = get_train_test_split(df, config)
        print(f"\nConfig: {config.name}")
        print(f"  Features ({len(config.all_features)}): {config.all_features}")
        print(f"  Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        print("  Train class distribution (normalized):")
        print(y_train.value_counts(normalize=True).round(3).sort_index())
        print("  Test class distribution (normalized):")
        print(y_test.value_counts(normalize=True).round(3).sort_index())
