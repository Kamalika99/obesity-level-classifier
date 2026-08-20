"""
train.py

Baseline model training and comparison for the Obesity Levels Classifier.

Trains Logistic Regression, Decision Tree, Random Forest, SVM, and XGBoost
on both feature configurations (full vs lifestyle-focused), using
Stratified 5-Fold Cross-Validation, then evaluates each on the held-out
test set.

Preprocessing is embedded inside each model's Pipeline so that fitting
happens only on training folds/data (no leakage).
"""

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from preprocessing import (
    FULL_CONFIG,
    LIFESTYLE_CONFIG,
    RANDOM_STATE,
    build_preprocessor,
    get_train_test_split,
    load_and_clean_data,
)

N_CV_FOLDS = 5


def get_baseline_models() -> dict:
    """Return a dict of {model_name: estimator} with reasonable baseline hyperparameters."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "SVM": SVC(probability=True, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def build_model_pipeline(preprocessor, estimator) -> Pipeline:
    """Wrap a preprocessor and estimator into a single sklearn Pipeline."""
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", estimator)])


def run_cross_validation(pipeline: Pipeline, X_train, y_train_encoded) -> dict:
    """Run Stratified 5-Fold CV and return mean CV metrics."""
    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro",
        "f1_weighted": "f1_weighted",
    }
    scores = cross_validate(
        pipeline, X_train, y_train_encoded, cv=cv, scoring=scoring, n_jobs=-1
    )
    return {
        "cv_accuracy": scores["test_accuracy"].mean(),
        "cv_precision_macro": scores["test_precision_macro"].mean(),
        "cv_recall_macro": scores["test_recall_macro"].mean(),
        "cv_f1_macro": scores["test_f1_macro"].mean(),
        "cv_f1_weighted": scores["test_f1_weighted"].mean(),
    }


def evaluate_on_test(pipeline: Pipeline, X_test, y_test_encoded) -> dict:
    """Fit-free evaluation on test set (pipeline must already be fit)."""
    y_pred = pipeline.predict(X_test)
    return {
        "test_accuracy": accuracy_score(y_test_encoded, y_pred),
        "test_precision_macro": precision_score(
            y_test_encoded, y_pred, average="macro", zero_division=0
        ),
        "test_recall_macro": recall_score(
            y_test_encoded, y_pred, average="macro", zero_division=0
        ),
        "test_f1_macro": f1_score(y_test_encoded, y_pred, average="macro"),
        "test_f1_weighted": f1_score(y_test_encoded, y_pred, average="weighted"),
    }


def run_experiment(df: pd.DataFrame, config, label_encoder: LabelEncoder) -> pd.DataFrame:
    """
    Run the full baseline model comparison for a given feature configuration.

    Returns a comparison DataFrame with CV and test metrics for every model.
    """
    X_train, X_test, y_train, y_test = get_train_test_split(df, config)
    y_train_enc = label_encoder.transform(y_train)
    y_test_enc = label_encoder.transform(y_test)

    results = []
    models = get_baseline_models()

    for name, estimator in models.items():
        t0 = time.time()
        preprocessor = build_preprocessor(config)
        pipeline = build_model_pipeline(preprocessor, estimator)

        cv_metrics = run_cross_validation(pipeline, X_train, y_train_enc)

        pipeline.fit(X_train, y_train_enc)
        test_metrics = evaluate_on_test(pipeline, X_test, y_test_enc)

        elapsed = time.time() - t0
        row = {"model": name, "feature_config": config.name, **cv_metrics, **test_metrics,
               "train_time_sec": round(elapsed, 2)}
        results.append(row)
        print(f"  [{config.name}] {name}: CV Macro-F1={cv_metrics['cv_f1_macro']:.4f} | "
              f"Test Macro-F1={test_metrics['test_f1_macro']:.4f} | {elapsed:.1f}s")

    return pd.DataFrame(results)


if __name__ == "__main__":
    df = load_and_clean_data("data/obesity_data.csv")

    label_encoder = LabelEncoder()
    label_encoder.fit(df["NObeyesdad"])

    print("\n=== Experiment A: Full feature set ===")
    results_full = run_experiment(df, FULL_CONFIG, label_encoder)

    print("\n=== Experiment B: Lifestyle-focused feature set (no Height/Weight) ===")
    results_lifestyle = run_experiment(df, LIFESTYLE_CONFIG, label_encoder)

    all_results = pd.concat([results_full, results_lifestyle], ignore_index=True)
    all_results = all_results.sort_values("cv_f1_macro", ascending=False)

    print("\n=== Full Model Comparison (sorted by CV Macro F1) ===")
    display_cols = ["model", "feature_config", "cv_accuracy", "cv_f1_macro",
                     "test_accuracy", "test_f1_macro", "test_f1_weighted"]
    print(all_results[display_cols].to_string(index=False))

    all_results.to_csv("reports/model_comparison.csv", index=False)
    print("\nSaved: reports/model_comparison.csv")
