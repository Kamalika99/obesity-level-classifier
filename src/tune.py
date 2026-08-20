"""
tune.py

Hyperparameter tuning for the top baseline models (Random Forest, XGBoost)
identified in the baseline comparison stage, run on both feature
configurations (full vs lifestyle-focused).

Uses RandomizedSearchCV (more efficient than exhaustive grid search given
the size of the hyperparameter spaces here) with Stratified 5-Fold CV,
optimizing for macro-F1 (appropriate for a balanced-but-not-huge,
multi-class problem where every class matters equally).
"""

import time

import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from preprocessing import (
    FULL_CONFIG,
    LIFESTYLE_CONFIG,
    RANDOM_STATE,
    build_preprocessor,
    get_train_test_split,
    load_and_clean_data,
)
from train import evaluate_on_test

N_CV_FOLDS = 5
N_ITER = 20  # RandomizedSearchCV iterations per model/config


def get_param_distributions() -> dict:
    """Hyperparameter search spaces for each tunable model, keyed to match
    the 'classifier__' prefix used inside the sklearn Pipeline."""
    return {
        "Random Forest": {
            "classifier__n_estimators": randint(100, 500),
            "classifier__max_depth": randint(4, 30),
            "classifier__min_samples_split": randint(2, 15),
            "classifier__min_samples_leaf": randint(1, 8),
            "classifier__max_features": ["sqrt", "log2", None],
        },
        "XGBoost": {
            "classifier__n_estimators": randint(100, 500),
            "classifier__max_depth": randint(3, 12),
            "classifier__learning_rate": uniform(0.01, 0.29),  # 0.01-0.30
            "classifier__subsample": uniform(0.6, 0.4),        # 0.6-1.0
            "classifier__colsample_bytree": uniform(0.6, 0.4), # 0.6-1.0
            "classifier__min_child_weight": randint(1, 10),
            "classifier__gamma": uniform(0, 0.5),
        },
    }


def get_tunable_models() -> dict:
    return {
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": XGBClassifier(
            eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=-1
        ),
    }


def tune_model(name, estimator, param_dist, preprocessor, X_train, y_train_enc):
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", estimator)])
    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=N_ITER,
        scoring="f1_macro",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    t0 = time.time()
    search.fit(X_train, y_train_enc)
    elapsed = time.time() - t0
    print(f"  {name}: best CV Macro-F1={search.best_score_:.4f} "
          f"({N_ITER} iters, {elapsed:.1f}s)")
    return search, elapsed


def run_tuning_experiment(df, config, label_encoder):
    X_train, X_test, y_train, y_test = get_train_test_split(df, config)
    y_train_enc = label_encoder.transform(y_train)
    y_test_enc = label_encoder.transform(y_test)

    models = get_tunable_models()
    param_dists = get_param_distributions()

    results = []
    fitted_searches = {}

    for name, estimator in models.items():
        preprocessor = build_preprocessor(config)
        search, elapsed = tune_model(
            name, estimator, param_dists[name], preprocessor, X_train, y_train_enc
        )
        test_metrics = evaluate_on_test(search.best_estimator_, X_test, y_test_enc)

        clean_params = {
            k.replace("classifier__", ""): v for k, v in search.best_params_.items()
        }

        results.append({
            "model": name,
            "feature_config": config.name,
            "cv_f1_macro": search.best_score_,
            "test_accuracy": test_metrics["test_accuracy"],
            "test_f1_macro": test_metrics["test_f1_macro"],
            "test_f1_weighted": test_metrics["test_f1_weighted"],
            "best_params": clean_params,
            "tune_time_sec": round(elapsed, 2),
        })
        fitted_searches[(name, config.name)] = search

    return pd.DataFrame(results), fitted_searches


def tune_single_job(model_name: str, config_name: str):
    """Tune exactly one (model, feature_config) pair and append its result
    to reports/tuning_results.csv. Designed to be run as its own short
    process so a single call fits comfortably inside a tool timeout."""
    import os

    df = load_and_clean_data("data/obesity_data.csv")
    label_encoder = LabelEncoder()
    label_encoder.fit(df["NObeyesdad"])
    joblib.dump(label_encoder, "models/label_encoder.joblib")

    config = FULL_CONFIG if config_name == "full" else LIFESTYLE_CONFIG
    X_train, X_test, y_train, y_test = get_train_test_split(df, config)
    y_train_enc = label_encoder.transform(y_train)
    y_test_enc = label_encoder.transform(y_test)

    models = get_tunable_models()
    param_dists = get_param_distributions()
    estimator = models[model_name]

    preprocessor = build_preprocessor(config)
    search, elapsed = tune_model(
        model_name, estimator, param_dists[model_name], preprocessor, X_train, y_train_enc
    )
    test_metrics = evaluate_on_test(search.best_estimator_, X_test, y_test_enc)

    def _to_native(v):
        if isinstance(v, np.floating):
            return float(v)
        if isinstance(v, np.integer):
            return int(v)
        return v

    clean_params = {
        k.replace("classifier__", ""): _to_native(v) for k, v in search.best_params_.items()
    }

    row = {
        "model": model_name,
        "feature_config": config_name,
        "cv_f1_macro": search.best_score_,
        "test_accuracy": test_metrics["test_accuracy"],
        "test_f1_macro": test_metrics["test_f1_macro"],
        "test_f1_weighted": test_metrics["test_f1_weighted"],
        "best_params": str(clean_params),
        "tune_time_sec": round(elapsed, 2),
    }

    out_csv = "reports/tuning_results.csv"
    row_df = pd.DataFrame([row])
    if os.path.exists(out_csv):
        existing = pd.read_csv(out_csv)
        combined = pd.concat([existing, row_df], ignore_index=True)
    else:
        combined = row_df
    combined.to_csv(out_csv, index=False)

    from model_io import save_pipeline
    kind = "xgboost" if model_name == "XGBoost" else "random_forest"
    prefix = f"models/{model_name.replace(' ', '_').lower()}_{config_name}"
    save_pipeline(search.best_estimator_, prefix, kind)
    print(f"Saved {prefix}_preprocessor.joblib + classifier ({kind})")
    print(f"Appended result row to {out_csv}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["Random Forest", "XGBoost"], required=False)
    parser.add_argument("--config", choices=["full", "lifestyle"], required=False)
    args = parser.parse_args()

    if args.model and args.config:
        tune_single_job(args.model, args.config)
    else:
        # Fallback: run everything sequentially (slow, not recommended interactively)
        df = load_and_clean_data("data/obesity_data.csv")
        label_encoder = LabelEncoder()
        label_encoder.fit(df["NObeyesdad"])

        all_results = []
        all_searches = {}
        for config in [FULL_CONFIG, LIFESTYLE_CONFIG]:
            print(f"\n=== Tuning on '{config.name}' feature config ===")
            res, searches = run_tuning_experiment(df, config, label_encoder)
            all_results.append(res)
            all_searches.update(searches)

        tuning_results = pd.concat(all_results, ignore_index=True)
        tuning_results = tuning_results.sort_values("cv_f1_macro", ascending=False)
        tuning_results.to_csv("reports/tuning_results.csv", index=False)
