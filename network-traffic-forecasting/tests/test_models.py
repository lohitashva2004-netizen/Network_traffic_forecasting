import numpy as np

from traffic_rf.config import FEATURES
from traffic_rf.data import generate_dataset
from traffic_rf.evaluation import regression_metrics
from traffic_rf.models import make_baselines, make_random_forest
from traffic_rf.preprocessing import split_data


def test_regression_metrics_perfect_and_known_values():
    y = np.array([1.0, 2.0, 3.0])
    assert regression_metrics(y, y) == {"rmse": 0.0, "mae": 0.0, "r2": 1.0}
    m = regression_metrics(y, y + 1)
    assert m["rmse"] == 1.0 and m["mae"] == 1.0


def test_random_forest_beats_baselines_on_test_set():
    df = generate_dataset()
    X_tr, X_te, y_tr, y_te = split_data(df)
    rf = regression_metrics(y_te, make_random_forest().fit(X_tr, y_tr).predict(X_te))
    assert rf["r2"] > 0.90
    for name, model in make_baselines().items():
        base = regression_metrics(y_te, model.fit(X_tr, y_tr).predict(X_te))
        assert rf["rmse"] < base["rmse"], name


def test_feature_importances_are_normalised_and_connections_dominate():
    df = generate_dataset()
    X_tr, _, y_tr, _ = split_data(df)
    imp = make_random_forest().fit(X_tr, y_tr).named_steps["model"].feature_importances_
    assert np.isclose(imp.sum(), 1.0)
    assert FEATURES[int(np.argmax(imp))] == "num_connections"


def test_training_is_reproducible():
    df = generate_dataset()
    X_tr, X_te, y_tr, _ = split_data(df)
    a = make_random_forest().fit(X_tr, y_tr).predict(X_te)
    b = make_random_forest().fit(X_tr, y_tr).predict(X_te)
    assert np.allclose(a, b)
