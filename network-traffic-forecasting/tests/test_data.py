import numpy as np

from traffic_rf.config import FEATURES, TARGET
from traffic_rf.data import generate_dataset


def test_shape_and_columns():
    df = generate_dataset()
    assert len(df) == 720
    assert set(FEATURES + [TARGET, "day"]) <= set(df.columns)


def test_is_deterministic():
    assert generate_dataset(seed=1).equals(generate_dataset(seed=1))
    assert not generate_dataset(seed=1).equals(generate_dataset(seed=2))


def test_value_ranges_match_documentation():
    df = generate_dataset()
    assert df["hour"].between(0, 23).all()
    assert df["avg_packet_size"].between(500, 1500).all()
    assert df["duration"].between(0.5, 10).all()
    assert df[TARGET].between(5, 80).all()
    assert set(df["protocol"].unique()) <= {0, 1}
    assert 0.1 < df["protocol"].mean() < 0.3  # ~20% UDP


def test_every_hour_appears_equally():
    counts = generate_dataset()["hour"].value_counts()
    assert counts.nunique() == 1 and counts.iloc[0] == 30
