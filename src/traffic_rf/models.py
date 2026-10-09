"""Model factories. Every model is a full Pipeline so cross-validation cannot leak."""
from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from .config import FEATURES, RF_PARAMS, SEED
from .preprocessing import build_preprocessor


def make_random_forest(features: list[str] = FEATURES, **overrides) -> Pipeline:
    params = {**RF_PARAMS, **overrides}
    return Pipeline(
        [("prep", build_preprocessor(features)), ("model", RandomForestRegressor(**params))]
    )


def make_baselines(features: list[str] = FEATURES) -> dict[str, Pipeline]:
    return {
        "Linear Regression": Pipeline(
            [("prep", build_preprocessor(features)), ("model", LinearRegression())]
        ),
        "Decision Tree": Pipeline(
            [
                ("prep", build_preprocessor(features)),
                ("model", DecisionTreeRegressor(max_depth=RF_PARAMS["max_depth"], random_state=SEED)),
            ]
        ),
    }
