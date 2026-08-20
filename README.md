# Obesity Levels Classifier — Model Training

Trains and evaluates a classifier that predicts a person's obesity category
(7 classes) from demographic, dietary, and lifestyle survey data. This
package contains only the modeling pipeline (data → features → training →
tuning → evaluation/explainability) — no application layer.

Dataset: UCI "Estimation of Obesity Levels Based On Eating Habits and
Physical Condition" (2,111 records, 17 columns; partially synthetically
augmented via SMOTE in its public form — see the analysis notebook, Stage 1,
for details and evidence).

## Project structure

```
obesity-model-training/
├── data/
│   └── obesity_data.csv              # raw dataset
├── notebooks/
│   └── obesity_classifier_analysis.ipynb   # full analysis: validation -> EDA -> training -> tuning -> explainability
├── src/
│   ├── data_validation.py            # validation checks (missing values, duplicates, invalid categories)
│   ├── preprocessing.py              # feature configs, leakage-safe pipelines, train/test split
│   ├── train.py                      # baseline training: 5 models x 2 feature configs, Stratified 5-fold CV
│   ├── tune.py                       # hyperparameter tuning (RandomizedSearchCV, per model/config)
│   └── model_io.py                   # cross-version-safe model save/load (see note below)
├── models/                           # trained model artifacts (see below)
├── reports/                          # all generated charts + result CSVs
└── README.md
```

## How to reproduce from scratch

```bash
pip install -r requirements.txt

# 1. Validate raw data (missing values, duplicates, invalid categories)
python src/data_validation.py

# 2. Sanity-check preprocessing + train/test split
python src/preprocessing.py

# 3. Train baselines: Logistic Regression, Decision Tree, Random Forest, SVM, XGBoost
#    on both feature configs (full / lifestyle-only) with Stratified 5-fold CV
python src/train.py
# -> reports/model_comparison.csv

# 4. Hyperparameter-tune the top 2 models (Random Forest, XGBoost) on both configs
python src/tune.py --model "XGBoost" --config full
python src/tune.py --model "Random Forest" --config full
python src/tune.py --model "XGBoost" --config lifestyle
python src/tune.py --model "Random Forest" --config lifestyle
# -> reports/tuning_results.csv, models/*_preprocessor.joblib, models/*_classifier.{json,joblib}

# 5. Full walkthrough with explainability (SHAP) and error analysis
jupyter notebook notebooks/obesity_classifier_analysis.ipynb
```

## Loading a trained model

```python
import sys
sys.path.append('src')
from model_io import load_pipeline

model = load_pipeline('models/xgboost_full', 'xgboost')   # final selected model
preds = model.predict(X)          # X must have the FULL_CONFIG feature columns
probs = model.predict_proba(X)
```

`src/model_io.py` exists because pickling a fitted `XGBClassifier` embeds a
raw internal booster buffer that can fail to deserialize
(`XGBoostError: input stream corrupted`) across different XGBoost
versions/platforms, or if the pickle file is corrupted/truncated in
transfer. XGBoost models here are saved via their own native `.json`
format instead; Random Forest models use joblib for the classifier and the
preprocessing `ColumnTransformer` (plain scikit-learn) is saved separately
in both cases.

## Modeling approach

- **Two feature configurations compared throughout**: `full` (includes
  Height/Weight) vs `lifestyle` (excludes them) — to separate "how much are
  we just re-deriving BMI" from "how much genuine lifestyle signal is
  there."
- **BMI itself is excluded from all models.** It's used only for exploratory
  analysis, since the target classes are themselves largely BMI-threshold
  defined — including it as a feature would be close to leaking the label
  definition into the input.
- **Leakage-safe pipelines**: all scaling/encoding is fit only inside
  cross-validation folds via `sklearn.Pipeline` + `ColumnTransformer`, never
  on the full dataset ahead of time.
- **Model selection criterion fixed in advance**: 5-fold stratified CV
  Macro-F1, chosen before looking at test-set numbers, to avoid
  cherry-picking whichever model happens to win on one particular split.

## Results

| Model | Feature set | CV Macro-F1 | Test Macro-F1 |
|---|---|---|---|
| **XGBoost (tuned, selected)** | full | **0.9577** | 0.9460 |
| Random Forest (tuned) | full | 0.9436 | 0.9704 |
| XGBoost (tuned) | lifestyle | 0.8378 | 0.8518 |
| Random Forest (tuned) | lifestyle | 0.8295 | 0.8286 |

Full detail — including baseline-vs-tuned comparisons, confusion matrix,
SHAP global/per-class/per-prediction explanations, and where the model's
errors concentrate — is in `notebooks/obesity_classifier_analysis.ipynb`
and `reports/`.

## Why accuracy is ~95% (and why that's expected, not a red flag)

The obesity class labels in this dataset are themselves largely defined by
BMI thresholds (BMI = Weight / Height²), and Weight/Height are included as
input features. So a large part of the "full" model's accuracy comes from
recovering a near-deterministic mathematical relationship, not from
discovering something subtle. This was explicitly tested for and reported
(see the "BMI investigation" and "Full vs lifestyle" sections of the
notebook) rather than left unexamined — the `lifestyle`-only accuracy
(~83-85%) is the more informative number for "how predictive are behavioral
factors alone."

## Limitations

- Dataset is partly synthetic (SMOTE-augmented), which likely makes classes
  more cleanly separable than real-world survey data would be — treat
  reported accuracy as an optimistic upper bound, not a real-world estimate.
- ~2,000 rows is small for a 7-class problem; hyperparameter tuning gains
  were modest and mixed in sign (see notebook Stage 4), consistent with
  limited data to reliably distinguish between hyperparameter settings.
- Not a medical diagnostic tool — reflects patterns in a specific survey
  dataset, not clinical obesity assessment.
