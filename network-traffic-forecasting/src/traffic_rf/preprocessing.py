"""Leakage-safe preprocessing: IQR clipping + Min-Max scaling fitted on training data only."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from .config import CLIP_FEATURES, FEATURES, SEED, TARGET, TEST_SIZE


class IQRClipper(BaseEstimator, TransformerMixin):
    """Clip selected columns to [Q1 - k*IQR, Q3 + k*IQR].

    Bounds are learned in ``fit`` (training data only) and re-used in ``transform``,
    so no information from the test set influences preprocessing.
    """

    def __init__(self, columns=None, k: float = 1.5):
        self.columns = columns
        self.k = k

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        cols = list(range(X.shape[1])) if self.columns is None else list(self.columns)
        if cols:
            q1, q3 = np.percentile(X[:, cols], [25, 75], axis=0)
        else:
            q1 = q3 = np.array([])
        iqr = q3 - q1
        self.columns_ = cols
        self.lower_ = q1 - self.k * iqr
        self.upper_ = q3 + self.k * iqr
        return self

    def transform(self, X):
        X = np.array(X, dtype=float, copy=True)
        if self.columns_:
            X[:, self.columns_] = np.clip(X[:, self.columns_], self.lower_, self.upper_)
        return X


def build_preprocessor(features: list[str] = FEATURES) -> Pipeline:
    """IQR clipping followed by Min-Max scaling to [0, 1]."""
    clip_idx = [features.index(c) for c in CLIP_FEATURES if c in features]
    return Pipeline([("clip", IQRClipper(columns=clip_idx)), ("scale", MinMaxScaler())])


def split_data(df: pd.DataFrame, features: list[str] = FEATURES):
    """Random 80/20 split (the protocol used in the original report)."""
    return train_test_split(
        df[features], df[TARGET], test_size=TEST_SIZE, random_state=SEED
    )


def chronological_split(df: pd.DataFrame, test_days: int = 3, features: list[str] = FEATURES):
    """Hold out the final ``test_days`` days: a stricter, forecasting-style evaluation."""
    cutoff = df["day"].max() - test_days
    train, test = df[df["day"] <= cutoff], df[df["day"] > cutoff]
    return train[features], test[features], train[TARGET], test[TARGET]
