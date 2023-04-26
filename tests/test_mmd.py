import os

import pytest

from modsim2.data.loader import DMPair
from modsim2.utils.config import load_configs

# project root = arc-model-similarites-phase-2/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
METRICS_CONFIG_PATH = os.path.join(PROJECT_ROOT, "tests", "testconfig", "metrics.yaml")


# Fixture that returns the metric config dictionary
@pytest.fixture(scope="module")
def metrics_config() -> dict:
    mmd_config = load_configs(METRICS_CONFIG_PATH)["metrics_config"]
    # filter down to only mmd configs
    mmd_config = {k: v for k, v in mmd_config.items() if v["class"] == "mmd"}
    return mmd_config


# This test checks that the distance between a dataset and itself is the expect value
# For this metric the expected value is always zero
def test_cifar_mmd_same(metrics_config: pytest.fixture):
    dmpair = DMPair(metrics_config=metrics_config)
    similarity_dict = dmpair.compute_similarity()
    similarity_dict_only_train = dmpair.compute_similarity(only_train=True)
    for k in metrics_config:
        assert similarity_dict[k] == 0
        assert similarity_dict_only_train[k] == 0


# This test checks that the distance between two different datasets is the expected
# value
# The calculated distance is checked against the known value for the seed - brittle
# test but useful for messing with code
def test_cifar_mmd_different(metrics_config: pytest.fixture):
    dmpair = DMPair(metrics_config=metrics_config, drop_percent_A=0.2, seed=42)
    similarity_dict = dmpair.compute_similarity()
    similarity_dict_only_train = dmpair.compute_similarity(only_train=True)
    for k in metrics_config:
        assert (
            similarity_dict[k] == metrics_config[k]["expected_results"]["diff_result"]
        )
        assert (
            similarity_dict_only_train[k]
            == metrics_config[k]["expected_results"]["diff_result_only_train"]
        )
