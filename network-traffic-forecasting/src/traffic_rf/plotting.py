"""Publication-style figures. Each function saves one PNG and returns its path."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

NAVY, BLUE, ORANGE, RED, GREY = "#1f3a5f", "#3b7dd8", "#e07a1f", "#c8433a", "#8a95a5"

plt.rcParams.update(
    {
        "figure.dpi": 130, "savefig.dpi": 160, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    }
)


def _save(fig, out_dir: Path, name: str) -> Path:
    path = Path(out_dir) / name
    fig.savefig(path)
    plt.close(fig)
    return path


def traffic_pattern(df, out):
    g = df.groupby("hour")["traffic_mbps"].agg(["mean", "std"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(g.index, g["mean"], color=NAVY, lw=2, label="Mean throughput")
    ax[0].fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"], color=BLUE, alpha=0.25, label="±1 SD")
    ax[0].set(title="Average daily traffic profile", xlabel="Hour of day", ylabel="Traffic (Mbps)")
    ax[0].legend()
    sc = ax[1].scatter(df["hour"], df["traffic_mbps"], c=df["num_connections"], cmap="Blues", s=14, alpha=0.7)
    ax[1].plot(g.index, g["mean"], color=RED, lw=2, label="Hourly mean")
    ax[1].set(title="All 720 hourly records", xlabel="Hour of day", ylabel="Traffic (Mbps)")
    ax[1].legend()
    fig.colorbar(sc, ax=ax[1], label="Connections")
    return _save(fig, out, "01_traffic_pattern.png")


def correlation_heatmap(df, out):
    cols = ["hour", "num_connections", "avg_packet_size", "protocol", "duration", "traffic_mbps"]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=40, ha="right")
    ax.set_yticks(range(len(cols)), cols)
    ax.grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    color="white" if abs(corr.iloc[i, j]) > 0.5 else "black", fontsize=8)
    ax.set_title("Pearson correlation matrix")
    fig.colorbar(im, ax=ax, shrink=0.8)
    return _save(fig, out, "02_correlation_heatmap.png")


def actual_vs_predicted(y_true, y_pred, metrics, out):
    err = np.abs(y_true - y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    sc = ax.scatter(y_true, y_pred, c=err, cmap="RdYlGn_r", s=24, alpha=0.85)
    lim = [min(y_true.min(), y_pred.min()) - 1, max(y_true.max(), y_pred.max()) + 1]
    ax.plot(lim, lim, "--", color=RED, label="Perfect fit")
    ax.set(xlim=lim, ylim=lim, xlabel="Actual traffic (Mbps)", ylabel="Predicted traffic (Mbps)",
           title=f"Actual vs predicted (RMSE {metrics['rmse']:.2f}, R² {metrics['r2']:.3f})")
    ax.legend()
    fig.colorbar(sc, ax=ax, label="|error| (Mbps)")
    return _save(fig, out, "03_actual_vs_predicted.png")


def feature_importance(names, impurity, perm_mean, perm_std, out):
    order = np.argsort(impurity)
    y = np.arange(len(names))
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    ax[0].barh(y, np.array(impurity)[order], color=NAVY)
    ax[0].set(title="Impurity-based importance", xlabel="Importance (sums to 1)")
    ax[1].barh(y, np.array(perm_mean)[order], xerr=np.array(perm_std)[order], color=BLUE)
    ax[1].set(title="Permutation importance (test set)", xlabel="Drop in R² when shuffled")
    ax[0].set_yticks(y, np.array(names)[order])
    for a, vals in zip(ax, (np.array(impurity)[order], np.array(perm_mean)[order])):
        for yi, v in zip(y, vals):
            a.text(v, yi, f" {v:.3f}", va="center", fontsize=8)
    return _save(fig, out, "04_feature_importance.png")


def forecast_window(actual, predicted, metrics, out):
    x = np.arange(len(actual))
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(x, actual, color=NAVY, lw=2, label="Actual")
    ax.plot(x, predicted, "--", color=RED, lw=2, label="Predicted")
    ax.fill_between(x, actual, predicted, color=RED, alpha=0.15, label="Error band")
    for d in range(1, len(actual) // 24):
        ax.axvline(d * 24, color=GREY, ls=":", lw=1)
    ax.set(xlabel="Hours into held-out window", ylabel="Traffic (Mbps)",
           title=f"Chronological hold-out: final 3 days (RMSE {metrics['rmse']:.2f}, R² {metrics['r2']:.3f})")
    ax.legend(ncol=3, loc="upper right")
    return _save(fig, out, "05_forecast_holdout.png")


def residual_diagnostics(y_true, y_pred, out):
    res = np.asarray(y_true) - np.asarray(y_pred)
    mu, sd = res.mean(), res.std()
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    ax[0, 0].scatter(y_pred, res, s=16, alpha=0.7, color=BLUE)
    ax[0, 0].axhline(0, color="k", ls="--", lw=1)
    ax[0, 0].set(title="Residuals vs predicted", xlabel="Predicted (Mbps)", ylabel="Residual (Mbps)")
    ax[0, 1].hist(res, bins=20, density=True, color=BLUE, alpha=0.8, edgecolor="white")
    xs = np.linspace(res.min(), res.max(), 200)
    ax[0, 1].plot(xs, stats.norm.pdf(xs, mu, sd), color=RED, lw=2, label=f"N({mu:.2f}, {sd:.2f}²)")
    ax[0, 1].set(title="Residual distribution", xlabel="Residual (Mbps)", ylabel="Density")
    ax[0, 1].legend()
    (osm, osr), (slope, icpt, r) = stats.probplot(res, dist="norm")
    ax[1, 0].scatter(osm, osr, s=14, color=BLUE)
    ax[1, 0].plot(osm, slope * osm + icpt, color=RED, label=f"R² = {r**2:.3f}")
    ax[1, 0].set(title="Normal Q-Q plot", xlabel="Theoretical quantiles", ylabel="Sample quantiles")
    ax[1, 0].legend()
    ax[1, 1].plot(res, color=BLUE, lw=0.9, marker="o", ms=2)
    ax[1, 1].axhline(0, color="k", ls="--", lw=1)
    for s in (2, -2):
        ax[1, 1].axhline(s * sd, color=RED, ls=":", lw=1)
    ax[1, 1].set(title="Residuals by test-sample index (±2σ dotted)", xlabel="Test sample", ylabel="Residual (Mbps)")
    fig.suptitle("Residual diagnostics", fontweight="bold")
    fig.tight_layout()
    return _save(fig, out, "06_residual_diagnostics.png"), {"mean": float(mu), "std": float(sd), "qq_r2": float(r**2)}


def error_by_range(y_true, y_pred, out):
    bins = [10, 20, 30, 40, 50, 60]
    labels = [f"{a}–{b}" for a, b in zip(bins[:-1], bins[1:])]
    df = pd.DataFrame({"abs_err": np.abs(np.asarray(y_true) - np.asarray(y_pred)),
                       "bin": pd.cut(np.asarray(y_true), bins, labels=labels)})
    groups = [df.loc[df["bin"] == l, "abs_err"].values for l in labels]
    keep = [(l, g) for l, g in zip(labels, groups) if len(g)]
    mae_all = df["abs_err"].mean()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].boxplot([g for _, g in keep], tick_labels=[l for l, _ in keep], patch_artist=True,
                  boxprops=dict(facecolor="#bcd3f0"), medianprops=dict(color=RED))
    ax[0].set(title="Absolute error by traffic range", xlabel="Actual traffic (Mbps)", ylabel="|error| (Mbps)")
    means = [g.mean() for _, g in keep]
    ax[1].bar([l for l, _ in keep], means, color=NAVY)
    ax[1].axhline(mae_all, color=RED, ls="--", label=f"Overall MAE = {mae_all:.2f}")
    for i, m in enumerate(means):
        ax[1].text(i, m, f"{m:.2f}", ha="center", va="bottom", fontsize=8)
    ax[1].set(title="Mean absolute error by traffic range", xlabel="Actual traffic (Mbps)", ylabel="MAE (Mbps)")
    ax[1].legend()
    return _save(fig, out, "07_error_by_range.png")


def partial_dependence_plot(pdp: dict, out):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, (name, (grid, avg)), c, xl in zip(
        ax, pdp.items(), (NAVY, RED), ("Number of connections", "Hour of day")
    ):
        a.plot(grid, avg, color=c, lw=2)
        a.set(title=f"Partial dependence: {name}", xlabel=xl, ylabel="Predicted traffic (Mbps)")
    return _save(fig, out, "08_partial_dependence.png")


def model_comparison(results: dict, out):
    models = list(results)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.8))
    colors = [GREY, ORANGE, NAVY]
    for a, key, title in zip(ax, ("rmse", "mae", "r2"), ("RMSE (Mbps) ↓", "MAE (Mbps) ↓", "R² ↑")):
        vals = [results[m][key] for m in models]
        a.bar(models, vals, color=colors[: len(models)])
        for i, v in enumerate(vals):
            a.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        a.set_title(title)
        a.tick_params(axis="x", labelrotation=15)
    return _save(fig, out, "09_model_comparison.png")


def cross_validation(cv: dict, out):
    k = np.arange(1, len(cv["rmse"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for a, key, c, yl in zip(ax, ("rmse", "r2"), (BLUE, NAVY), ("RMSE (Mbps)", "R²")):
        a.bar(k, cv[key], color=c)
        a.axhline(cv[key].mean(), color=RED, ls="--", label=f"Mean = {cv[key].mean():.3f} (SD {cv[key].std():.3f})")
        a.set(xlabel="Fold", ylabel=yl, title=f"{yl} per fold (10-fold CV)", xticks=k)
        a.set_ylim(0, max(cv[key]) * 1.3 if key == "rmse" else 1.12)
        a.legend(loc="upper center", fontsize=8)
    return _save(fig, out, "10_cross_validation.png")


def learning_curve_plot(sizes, tr, va, out):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for arr, c, lab in ((tr, NAVY, "Training RMSE"), (va, RED, "Validation RMSE (5-fold CV)")):
        m, s = arr.mean(1), arr.std(1)
        ax.plot(sizes, m, "o-", color=c, label=lab)
        ax.fill_between(sizes, m - s, m + s, color=c, alpha=0.15)
    ax.set(xlabel="Training set size", ylabel="RMSE (Mbps)", title="Learning curve")
    ax.legend()
    return _save(fig, out, "11_learning_curve.png")


def hyperparameter_sensitivity(trees, depth, out):
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    for row, (vals, rmse, r2, name, chosen) in zip(ax, (trees, depth)):
        xs = np.arange(len(vals))
        labels = [str(v) for v in vals]
        for a, y, c, yl in zip(row, (rmse, r2), (RED, NAVY), ("CV RMSE (Mbps)", "CV R²")):
            a.plot(xs, y, "o-", color=c)
            a.set_xticks(xs, labels)
            if chosen in vals:
                a.axvline(vals.index(chosen), color=GREY, ls="--", label=f"Chosen = {chosen}")
                a.legend()
            a.set(xlabel=name, ylabel=yl, title=f"{yl} vs {name}")
    fig.tight_layout()
    return _save(fig, out, "12_hyperparameter_sensitivity.png")


def ablation_plot(results: dict, out):
    labels = list(results)
    r2 = [results[l]["r2"] for l in labels]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.barh(labels, r2, color=[NAVY if l.startswith("All") else BLUE for l in labels])
    for i, v in enumerate(r2):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=9)
    ax.set(xlim=(0.75, 0.95), xlabel="Test R²", title="Feature ablation: how much does each feature set explain?")
    ax.invert_yaxis()
    return _save(fig, out, "13_feature_ablation.png")
