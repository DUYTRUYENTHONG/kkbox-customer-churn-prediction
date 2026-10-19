"""Fit an interpretable baseline and report held-out model metrics."""

from __future__ import annotations

from pathlib import Path
from math import ceil

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_baseline(features: pd.DataFrame, model_path: Path) -> dict[str, float | int | str]:
    """Train once and evaluate on a stratified held-out sample."""
    if features["msno"].duplicated().any():
        raise ValueError("Expected one row per msno")
    if set(features["is_churn"].unique()) != {0, 1}:
        raise ValueError("Labels must contain both 0 and 1")
    if len(features) < 20 or features["is_churn"].value_counts().min() < 4:
        raise ValueError("Need at least 20 users and four examples of each class")

    labels = features["is_churn"]
    data = features.drop(columns=["msno", "is_churn"])
    model = Pipeline(
        [
            ("fill", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.25, stratify=labels, random_state=42
    )
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    prevalence = float(y_train.mean())
    top_count = max(1, ceil(len(y_test) * 0.1))
    top_labels = y_test.iloc[probabilities.argsort()[-top_count:]]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return {
        "evaluation": "stratified random holdout within one label cohort; not an out-of-time test",
        "users": len(features),
        "train_users": len(y_train),
        "holdout_users": len(y_test),
        "train_churn_rate": round(prevalence, 6),
        "holdout_churn_rate": round(float(y_test.mean()), 6),
        "prevalence_log_loss": round(float(log_loss(y_test, [prevalence] * len(y_test), labels=[0, 1])), 6),
        "holdout_log_loss": round(float(log_loss(y_test, probabilities)), 6),
        "holdout_auc_roc": round(float(roc_auc_score(y_test, probabilities)), 6),
        "holdout_average_precision": round(float(average_precision_score(y_test, probabilities)), 6),
        "holdout_brier_score": round(float(brier_score_loss(y_test, probabilities)), 6),
        "top_decile_churn_rate": round(float(top_labels.mean()), 6),
        "top_decile_users": top_count,
    }
