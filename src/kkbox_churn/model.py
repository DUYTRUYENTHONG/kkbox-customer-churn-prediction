"""Fit an interpretable baseline and report held-out model metrics."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def train_baseline(features: pd.DataFrame, model_path: Path) -> dict[str, float | int]:
    """Train once and evaluate on a stratified held-out sample."""
    if features["msno"].duplicated().any():
        raise ValueError("Expected one row per msno")
    if len(features) < 20 or features["is_churn"].value_counts().min() < 2:
        raise ValueError("Need at least 20 users and two examples of each class")
    if set(features["is_churn"].unique()) != {0, 1}:
        raise ValueError("Labels must contain both 0 and 1")

    labels = features["is_churn"]
    data = features.drop(columns=["msno", "is_churn"])
    categorical = ["city", "registered_via"]
    numeric = [column for column in data if column not in categorical]
    processor = ColumnTransformer(
        [
            ("numeric", Pipeline([("fill", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("category", Pipeline([("fill", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ]
    )
    model = Pipeline([("features", processor), ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.25, stratify=labels, random_state=42
    )
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return {
        "users": len(features),
        "holdout_users": len(y_test),
        "holdout_log_loss": round(float(log_loss(y_test, probabilities)), 6),
        "holdout_auc_roc": round(float(roc_auc_score(y_test, probabilities)), 6),
    }

