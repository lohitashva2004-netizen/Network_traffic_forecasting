# Methodology

## Problem formulation
Supervised regression: given five hourly network statistics, predict aggregate throughput (Mbps) for that hour.

## Data generation
`generate_dataset()` builds 30 days × 24 hours of records with `numpy.random.RandomState(42)`:

- Base load: `20 + 30·sin²((hour − 8)·π/12)` plus N(0, 3²) noise, clipped to [5, 80] Mbps.
- `num_connections = traffic × U(8, 12)`.
- `avg_packet_size ~ U(500, 1500)`, `duration ~ U(0.5, 10)`, `protocol ~ Bernoulli(0.2)`.

## Preprocessing
Implemented as a scikit-learn `Pipeline` so that every statistic is learned from training folds only:

1. **IQRClipper** (custom transformer): clips `num_connections`, `avg_packet_size`, `duration` to `[Q1 − 1.5·IQR, Q3 + 1.5·IQR]`. Bounds come from the training data and are re-applied to unseen data. `hour` (cyclic) and `protocol` (binary) are not clipped.
2. **MinMaxScaler** to [0, 1]. Tree models are scale-invariant, so this is retained mainly for pipeline consistency and for fair comparison with the linear baseline.

## Model
Random Forest regression: 100 bootstrapped trees, `max_depth=10`, all other settings at scikit-learn defaults (notably `max_features=1.0`, i.e. every feature is considered at each split, and `min_samples_leaf=1`). Predictions are the mean over trees. Splits minimise squared-error impurity.

## Evaluation protocol
| Experiment | Purpose |
|---|---|
| Random 80/20 split (seed 42) | Headline test metrics |
| 10-fold CV on training set | Sensitivity to the particular split |
| Chronological hold-out (last 3 days) | Forecasting-style generalisation |
| Learning curve (5-fold CV) | Bias/variance diagnosis |
| Hyperparameter sweeps (5-fold CV, training set only) | Sensitivity of `n_estimators`, `max_depth`; the test set is never used for tuning |
| Residual diagnostics | Bias, heteroscedasticity, normality |
| Partial dependence | Learned marginal effects |
| Impurity + permutation importance | Model-internal vs. model-agnostic attribution |
| Feature ablation | Unique vs. overlapping information |

### Metrics
- RMSE = √(mean((y − ŷ)²)), in Mbps; penalises large errors.
- MAE = mean(|y − ŷ|), in Mbps.
- R² = 1 − SS_res / SS_tot.

## Reproducibility
All randomness is seeded (`SEED = 42` in `config.py`). `python scripts/run_experiments.py` regenerates `data/`, `results/metrics.json` and all figures; `pytest` checks determinism and key invariants.
