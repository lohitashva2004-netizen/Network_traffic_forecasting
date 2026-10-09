"""Metrics and experiment helpers (cross-validation, sweeps, ablations)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, learning_curve

from .config import SEED
from .models import make_random_forest
from .preprocessing import split_data

_SCORING = {"rmse": "neg_root_mean_squared_error", "r2": "r2"}


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def cross_validate_rf(X, y, n_splits: int = 10) -> dict[str, np.ndarray]:
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    res = cross_validate(make_random_forest(), X, y, cv=cv, scoring=_SCORING)
    return {"rmse": -res["test_rmse"], "r2": res["test_r2"]}


def sweep_hyperparameter(X, y, name: str, values, n_splits: int = 5):
    """5-fold CV (training set only) over one Random Forest hyperparameter."""
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    rmse, r2 = [], []
    for v in values:
        res = cross_validate(make_random_forest(**{name: v}), X, y, cv=cv, scoring=_SCORING)
        rmse.append((-res["test_rmse"]).mean())
        r2.append(res["test_r2"].mean())
    return np.array(rmse), np.array(r2)


def rf_learning_curve(X, y, n_splits: int = 5):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    sizes, tr, va = learning_curve(
        make_random_forest(), X, y, cv=cv,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring="neg_root_mean_squared_error",
    )
    return sizes, -tr, -va


def ablation_study(df, feature_sets: dict[str, list[str]]):
    """Test-set metrics for several feature subsets (same random split as the main run)."""
    out = {}
    for label, feats in feature_sets.items():
        X_tr, X_te, y_tr, y_te = split_data(df, feats)
        model = make_random_forest(feats).fit(X_tr, y_tr)
        out[label] = regression_metrics(y_te, model.predict(X_te))
    return out
