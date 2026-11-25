"""Model pipelines (all preprocessing and resampling is fitted on training data only) and metrics."""
from __future__ import annotations

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .data import CATEGORICAL, NUMERIC


def preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ], sparse_threshold=0)


def xgboost(pos_weight: float, device: str = "cpu", n_estimators: int = 250, seed: int = 0):
    """Settings from a 40-configuration random search scored on the 1995 validation year only: shallow trees and
    a small learning rate won, because the fraud signal is weak and deeper trees memorise noise."""
    from xgboost import XGBClassifier

    return XGBClassifier(n_estimators=n_estimators, learning_rate=0.01, max_depth=3, min_child_weight=1,
                         subsample=1.0, colsample_bytree=0.8, reg_lambda=5.0, scale_pos_weight=1.0,
                         eval_metric="aucpr", tree_method="hist", device=device, random_state=seed)


def candidates(pos_weight: float, device: str = "cpu", seed: int = 0) -> dict:
    def wrap(model):
        return Pipeline([("prep", preprocessor()), ("model", model)])

    return {
        "Decision Tree + SMOTE (original best)": ImbPipeline([
            ("prep", preprocessor()), ("smote", SMOTE(random_state=seed)),
            ("model", DecisionTreeClassifier(max_depth=8, min_samples_leaf=20, random_state=seed))]),
        "Logistic Regression (class-weighted)": wrap(LogisticRegression(max_iter=3000, class_weight="balanced")),
        "Random Forest (class-weighted)": wrap(RandomForestClassifier(n_estimators=500, min_samples_leaf=5,
                                                                      class_weight="balanced_subsample", n_jobs=-1,
                                                                      random_state=seed)),
        "XGBoost": wrap(xgboost(pos_weight, device, seed=seed)),
    }


def best_f1_threshold(y, scores) -> float:
    p, r, t = precision_recall_curve(y, scores)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    return float(t[int(np.argmax(f1))])


def fraud_caught_when_reviewing(y, scores, share: float) -> float:
    """Share of all fraud found if investigators review only the `share` of claims with the highest scores."""
    y, scores = np.asarray(y), np.asarray(scores)
    k = max(int(round(share * len(y))), 1)
    top = np.argsort(-scores, kind="stable")[:k]
    return float(y[top].sum() / max(y.sum(), 1))


def metrics(y, scores, threshold: float) -> dict:
    y = np.asarray(y)
    flagged = np.asarray(scores) >= threshold
    return {k: round(float(v), 4) for k, v in {
        "pr_auc": average_precision_score(y, scores),
        "roc_auc": roc_auc_score(y, scores),
        "precision": precision_score(y, flagged, zero_division=0),
        "recall": recall_score(y, flagged),
        "f1": f1_score(y, flagged),
        "fraud_caught_reviewing_10pct": fraud_caught_when_reviewing(y, scores, 0.10),
        "fraud_caught_reviewing_20pct": fraud_caught_when_reviewing(y, scores, 0.20),
        "threshold": threshold,
    }.items()}


def bootstrap_ci(y, scores, metric, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% confidence interval by resampling the test claims: with about 200 fraud cases in a test year,
    differences of a few hundredths between models are within noise."""
    y, scores = np.asarray(y), np.asarray(scores)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() != y[idx].max():
            values.append(metric(y[idx], scores[idx]))
    lo, hi = np.quantile(values, [0.025, 0.975])
    return round(float(lo), 4), round(float(hi), 4)
