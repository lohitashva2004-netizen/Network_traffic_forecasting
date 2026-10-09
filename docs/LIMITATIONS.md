# Limitations and threats to validity

Being explicit about these is part of doing the analysis properly.

## 1. Synthetic data
All results come from a simulator, not measured traffic. They demonstrate a correct, leak-free modelling workflow; they do **not** estimate real-world accuracy. Real traffic has bursts, weekly seasonality, holidays, failures and long-range dependence that this generator does not model.

## 2. Target-derived feature
`num_connections` is generated as `traffic × U(8, 12)`, so it is partly a noisy transformation of the target. In a real deployment, connection counts and throughput are related but neither is a deterministic function of the other. This inflates the feature's importance and the model's headline accuracy. The feature ablation (`results/figures/13_feature_ablation.png`) is included to put the importance scores in context: `hour` alone reaches R² = 0.902, versus 0.864 for `num_connections` alone.

## 3. Stylised daily pattern
The base profile `sin²((hour − 8)π/12)` has a 12-hour period, producing peaks near 02:00 and 14:00 and troughs near 08:00 and 20:00 in this simulation. Real campuses typically show daytime-heavy or evening-heavy profiles; the shape here is a convenient test signal, not an empirical finding.

## 4. Independence of records
Rows are hourly snapshots from the same 30 days. Adjacent hours are correlated, so a random split is optimistic. The chronological hold-out reduces, but does not eliminate, this concern at only 30 days of data.

## 5. Small sample
With 720 rows (144 test), metric estimates carry noticeable uncertainty; the 10-fold CV standard deviation (RMSE ± 0.27 Mbps) gives a sense of it.

## 6. Mild residual bias
Mean test residual is −0.58 Mbps (slight over-prediction). It is small relative to the error spread, but it is not zero, and I would not describe the residuals as perfectly centred.

## 7. Not a time-series model
No lag, rolling or calendar features are used, so the model cannot exploit autocorrelation. It predicts each hour from that hour's own statistics.
