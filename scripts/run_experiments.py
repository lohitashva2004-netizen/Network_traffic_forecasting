#!/usr/bin/env python
"""Run the full experiment suite: python scripts/run_experiments.py"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traffic_rf.pipeline import run  # noqa: E402

if __name__ == "__main__":
    metrics = run(output_dir=ROOT / "results", data_dir=ROOT / "data")
    print(json.dumps({k: metrics[k] for k in ("random_forest_test", "chronological_holdout_last_3_days")}, indent=2))
    print(f"Figures written to {ROOT / 'results' / 'figures'}")
