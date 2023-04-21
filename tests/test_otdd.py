import os

from modsim2.data.loader import DMPair
from modsim2.utils.config import load_configs

# project root = arc-model-similarites-phase-2/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Get metric config
METRICS_CONFIG_PATH = os.path.join(PROJECT_ROOT, "tests", "testconfig", "metrics.yaml")
CONFIGS = load_configs(METRICS_CONFIG_PATH)
METRIC_CONFIG = CONFIGS["metric_config"]
METRIC_CONFIG = {k: v for k, v in METRIC_CONFIG.items() if v["class"] == "otdd"}


# This test checks that the distance between a dataset and itself is the expect value
# When exact calculations are used then this will be zero
# Approximate methods may be non-zero and these are checked against the known value
# for the seed
def test_cifar_otdd_same():
    dmpair = DMPair(metric_config=METRIC_CONFIG, seed=42)
    similarity_dict = dmpair.compute_similarity(only_train=False, metric_seed=42)
    for k in similarity_dict:
        assert similarity_dict[k] == METRIC_CONFIG[k]["expected_results"]["same_result"]
    similarity_dict = dmpair.compute_similarity(only_train=True)
    for k in similarity_dict:
        assert (
            similarity_dict[k]
            == METRIC_CONFIG[k]["expected_results"]["same_result_only_train"]
        )


# This test checks that the distance between a dataset and itself is the expect value
# When exact calculations are used then this will be zero
# Approximate methods may be non-zero and these are checked against the known value
# for the seed
def test_cifar_otdd_different():
    dmpair = DMPair(metric_config=METRIC_CONFIG, drop_percent_A=0.2, seed=42)
    similarity_dict = dmpair.compute_similarity(only_train=False, metric_seed=42)
    for k in similarity_dict:
        assert similarity_dict[k] == METRIC_CONFIG[k]["expected_results"]["diff_result"]
    similarity_dict = dmpair.compute_similarity(only_train=True, metric_seed=42)
    for k in similarity_dict:
        assert (
            similarity_dict[k]
            == METRIC_CONFIG[k]["expected_results"]["diff_result_only_train"]
        )
