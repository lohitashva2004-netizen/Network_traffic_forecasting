"""Synthetic campus-network traffic generator.

The generator reproduces the dataset described in the project report:
30 days x 24 hours = 720 hourly records with five input features and the
aggregate throughput (Mbps) as the regression target.

Note
----
``num_connections`` is derived from the traffic level (connections ~ traffic x U(8, 12)).
This mirrors the original coursework design, but it means the feature carries
direct information about the target. See ``docs/LIMITATIONS.md`` and the
ablation study for how this affects the conclusions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import N_DAYS, SEED


def generate_dataset(n_days: int = N_DAYS, seed: int = SEED) -> pd.DataFrame:
    """Generate the hourly campus traffic dataset (deterministic for a given seed)."""
    rng = np.random.RandomState(seed)
    n = n_days * 24

    hours = np.tile(np.arange(24), n_days)
    base = 20 + 30 * np.sin((hours - 8) * np.pi / 12) ** 2
    traffic = np.clip(base + rng.normal(0, 3, n), 5, 80)
    num_conns = traffic * rng.uniform(8, 12, n)
    avg_pkt_size = rng.uniform(500, 1500, n)
    protocol = rng.choice([0, 1], n, p=[0.8, 0.2])
    duration = rng.uniform(0.5, 10, n)

    return pd.DataFrame(
        {
            "day": np.repeat(np.arange(1, n_days + 1), 24),
            "hour": hours,
            "num_connections": num_conns,
            "avg_packet_size": avg_pkt_size,
            "protocol": protocol,
            "duration": duration,
            "traffic_mbps": traffic,
        }
    )
