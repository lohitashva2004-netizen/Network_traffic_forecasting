"""Central configuration: feature names, seeds, and model hyperparameters."""

SEED = 42
N_DAYS = 30
TEST_SIZE = 0.20

FEATURES = ["hour", "num_connections", "avg_packet_size", "protocol", "duration"]
TARGET = "traffic_mbps"

# Features that receive IQR outlier clipping (protocol is categorical, hour is cyclic).
CLIP_FEATURES = ["num_connections", "avg_packet_size", "duration"]

RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": SEED,
    "n_jobs": -1,
}
