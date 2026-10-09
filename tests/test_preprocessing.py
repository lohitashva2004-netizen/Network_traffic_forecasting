import numpy as np
import pytest

from traffic_rf.config import FEATURES
from traffic_rf.data import generate_dataset
from traffic_rf.preprocessing import IQRClipper, chronological_split, split_data


def test_split_sizes_and_disjoint():
    df = generate_dataset()
    X_tr, X_te, y_tr, y_te = split_data(df)
    assert (len(X_tr), len(X_te)) == (576, 144)
    assert set(X_tr.index).isdisjoint(X_te.index)


def test_iqr_clipper_clips_outliers_only():
    x = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [1000.0]])
    out = IQRClipper(columns=[0]).fit(x).transform(x)
    assert out.max() < 1000.0
    assert np.allclose(out[:5], x[:5])


def test_iqr_clipper_uses_training_bounds_only():
    train = np.arange(100, dtype=float).reshape(-1, 1)
    clipper = IQRClipper(columns=[0]).fit(train)
    test = np.array([[10_000.0]])
    assert clipper.transform(test)[0, 0] == pytest.approx(clipper.upper_[0])


def test_clipper_leaves_other_columns_untouched():
    x = np.column_stack([np.arange(10.0), np.arange(10.0) * 1000])
    x[0, 1] = -1e9
    out = IQRClipper(columns=[0]).fit(x).transform(x)
    assert out[0, 1] == -1e9


def test_chronological_split_has_no_time_overlap():
    df = generate_dataset()
    X_tr, X_te, _, _ = chronological_split(df, test_days=3)
    assert len(X_te) == 72
    assert df.loc[X_tr.index, "day"].max() < df.loc[X_te.index, "day"].min()
