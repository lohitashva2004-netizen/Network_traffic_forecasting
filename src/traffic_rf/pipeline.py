"""End-to-end experiment: data -> training -> evaluation -> figures + metrics.json."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.inspection import partial_dependence, permutation_importance

from . import plotting as P
from .config import FEATURES, RF_PARAMS, SEED
from .data import generate_dataset
from .evaluation import (ablation_study, cross_validate_rf, regression_metrics,
                         rf_learning_curve, sweep_hyperparameter)
from .models import make_baselines, make_random_forest
from .preprocessing import chronological_split, split_data


def run(output_dir: str | Path = "results", data_dir: str | Path = "data") -> dict:
    out = Path(output_dir)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    # 1. Data ----------------------------------------------------------------
    df = generate_dataset()
    df.to_csv(Path(data_dir) / "campus_traffic_hourly.csv", index=False)
    X_tr, X_te, y_tr, y_te = split_data(df)
    P.traffic_pattern(df, fig_dir)
    P.correlation_heatmap(df[["hour", "num_connections", "avg_packet_size", "protocol", "duration", "traffic_mbps"]], fig_dir)

    # 2. Main model ----------------------------------------------------------
    rf = make_random_forest().fit(X_tr, y_tr)
    pred = rf.predict(X_te)
    rf_metrics = regression_metrics(y_te, pred)
    P.actual_vs_predicted(y_te.values, pred, rf_metrics, fig_dir)

    # 3. Feature importance (impurity + permutation) -------------------------
    impurity = rf.named_steps["model"].feature_importances_
    perm = permutation_importance(rf, X_te, y_te, n_repeats=20, random_state=SEED, scoring="r2")
    P.feature_importance(FEATURES, impurity, perm.importances_mean, perm.importances_std, fig_dir)

    # 4. Chronological hold-out (final 3 days) -------------------------------
    Xc_tr, Xc_te, yc_tr, yc_te = chronological_split(df)
    rf_c = make_random_forest().fit(Xc_tr, yc_tr)
    pred_c = rf_c.predict(Xc_te)
    chrono_metrics = regression_metrics(yc_te, pred_c)
    P.forecast_window(yc_te.values, pred_c, chrono_metrics, fig_dir)

    # 5. Diagnostics ---------------------------------------------------------
    _, resid = P.residual_diagnostics(y_te.values, pred, fig_dir)
    P.error_by_range(y_te.values, pred, fig_dir)
    pdp = {}
    for feat in ("num_connections", "hour"):
        r = partial_dependence(rf, X_tr.astype(float), [feat], grid_resolution=50, kind="average")
        pdp[feat] = (r["grid_values"][0], r["average"][0])
    P.partial_dependence_plot(pdp, fig_dir)

    # 6. Baselines -----------------------------------------------------------
    comparison = {"Linear Regression": None, "Decision Tree": None}
    for name, model in make_baselines().items():
        comparison[name] = regression_metrics(y_te, model.fit(X_tr, y_tr).predict(X_te))
    comparison["Random Forest"] = rf_metrics
    P.model_comparison(comparison, fig_dir)

    # 7. Cross-validation, learning curve, sweeps (training data only) -------
    cv = cross_validate_rf(X_tr, y_tr, n_splits=10)
    P.cross_validation(cv, fig_dir)
    sizes, tr_c, va_c = rf_learning_curve(X_tr, y_tr)
    P.learning_curve_plot(sizes, tr_c, va_c, fig_dir)
    tree_vals, depth_vals = [5, 10, 20, 30, 50, 100, 200], [1, 2, 3, 5, 7, 10, 15, 20, None]
    t_rmse, t_r2 = sweep_hyperparameter(X_tr, y_tr, "n_estimators", tree_vals)
    d_rmse, d_r2 = sweep_hyperparameter(X_tr, y_tr, "max_depth", depth_vals)
    P.hyperparameter_sensitivity(
        (tree_vals, t_rmse, t_r2, "n_estimators", 100),
        (depth_vals, d_rmse, d_r2, "max_depth", 10), fig_dir)

    # 8. Ablation ------------------------------------------------------------
    no_conn = [f for f in FEATURES if f != "num_connections"]
    ablation = ablation_study(df, {
        "All five features": FEATURES,
        "Without num_connections": no_conn,
        "hour only": ["hour"],
        "num_connections only": ["num_connections"],
    })
    P.ablation_plot(ablation, fig_dir)

    # 9. Persist -------------------------------------------------------------
    metrics = {
        "config": {"seed": SEED, "n_records": len(df), "n_train": len(X_tr), "n_test": len(X_te), "rf_params": RF_PARAMS},
        "random_forest_test": rf_metrics,
        "chronological_holdout_last_3_days": chrono_metrics,
        "baselines_test": comparison,
        "cross_validation_10fold": {
            "rmse_mean": float(cv["rmse"].mean()), "rmse_std": float(cv["rmse"].std()),
            "r2_mean": float(cv["r2"].mean()), "r2_std": float(cv["r2"].std())},
        "feature_importance": {
            "impurity": dict(zip(FEATURES, map(float, impurity))),
            "permutation_r2_drop": dict(zip(FEATURES, map(float, perm.importances_mean)))},
        "residuals": resid,
        "ablation_test": ablation,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    m = run()
    print(json.dumps(m["random_forest_test"], indent=2))
