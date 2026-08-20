"""
model_io.py

Robust, cross-version-safe saving/loading for the trained pipelines.

XGBoost models are saved using XGBoost's own native format (JSON), not raw
pickle, because pickling an XGBClassifier embeds a raw booster byte-buffer
that can fail to deserialize ("input stream corrupted") when read back with
a different XGBoost build, or when the pickle file gets corrupted/truncated
in transfer (e.g. partial sync via cloud-storage placeholder files). Native
XGBoost format is explicitly designed to be portable across versions/OSes.

The preprocessing ColumnTransformer is plain scikit-learn and is saved with
joblib as usual, which is robust for sklearn-only objects.

A lightweight PipelineBundle wraps both pieces back together and exposes the
same interface app.py/notebooks expect (predict, predict_proba, named_steps).
"""

from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


class PipelineBundle:
    """Minimal drop-in replacement for a fitted sklearn Pipeline with steps
    ('preprocessor', 'classifier'), so existing code that does
    pipeline.predict(...) / pipeline.named_steps['classifier'] keeps working
    regardless of how the classifier was persisted."""

    def __init__(self, preprocessor, classifier):
        self.preprocessor = preprocessor
        self.classifier = classifier
        self.named_steps = {"preprocessor": preprocessor, "classifier": classifier}

    def predict(self, X):
        return self.classifier.predict(self.preprocessor.transform(X))

    def predict_proba(self, X):
        return self.classifier.predict_proba(self.preprocessor.transform(X))


def save_pipeline(pipeline, path_prefix: str, model_kind: str):
    """
    Save a fitted sklearn Pipeline(preprocessor, classifier).

    model_kind: 'xgboost' or 'random_forest'. XGBoost classifiers are saved
    natively; everything else falls back to joblib.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    joblib.dump(preprocessor, f"{path_prefix}_preprocessor.joblib")

    if model_kind == "xgboost":
        classifier.save_model(f"{path_prefix}_classifier.json")
    else:
        joblib.dump(classifier, f"{path_prefix}_classifier.joblib")


def load_pipeline(path_prefix: str, model_kind: str) -> PipelineBundle:
    """Load a pipeline saved with save_pipeline() back into a PipelineBundle."""
    preprocessor = joblib.load(f"{path_prefix}_preprocessor.joblib")

    if model_kind == "xgboost":
        classifier = XGBClassifier()
        classifier.load_model(f"{path_prefix}_classifier.json")
    else:
        classifier = joblib.load(f"{path_prefix}_classifier.joblib")

    return PipelineBundle(preprocessor, classifier)
