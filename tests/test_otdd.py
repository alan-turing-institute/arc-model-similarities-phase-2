import os

import pytest

from modsim2.data.loader import DMPair
from modsim2.utils.config import load_configs

# project root = arc-model-similarites-phase-2/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Get metric config
METRICS_CONFIG_PATH = os.path.join(PROJECT_ROOT, "tests", "testconfig", "metrics.yaml")


# Fixture that yields the metric config dictionary
@pytest.fixture(scope="module")
def metrics_config():
    otdd_config = load_configs(METRICS_CONFIG_PATH)["metrics_config"]
    # filter down to only mmd configs
    otdd_config = {k: v for k, v in otdd_config.items() if v["class"] == "otdd"}
    yield otdd_config


# This test checks that the distance between a dataset and itself is the expect value
# When exact calculations are used then this will be zero
# Approximate methods may be non-zero and these are checked against the known value
# for the seed
def test_cifar_otdd_same(metrics_config):
    dmpair = DMPair(metrics_config=metrics_config, seed=42)

    similarity_dict = dmpair.compute_similarity(only_train=False)
    similarity_dict_only_train = dmpair.compute_similarity(only_train=True)
    for k in metrics_config:
        assert (
            similarity_dict[k] == metrics_config[k]["expected_results"]["same_result"]
        )
        assert (
            similarity_dict_only_train[k]
            == metrics_config[k]["expected_results"]["same_result_only_train"]
        )


# This test checks that the distance between two different datasets is the expected
# value
# The calculated distance is checked against the known value for the seed - brittle
# test but useful for messing with code
def test_cifar_otdd_different(metrics_config):
    dmpair = DMPair(metrics_config=metrics_config, drop_percent_A=0.2, seed=42)
    similarity_dict = dmpair.compute_similarity(only_train=False)
    similarity_dict_only_train = dmpair.compute_similarity(only_train=True)
    for k in metrics_config:
        assert (
            similarity_dict[k] == metrics_config[k]["expected_results"]["diff_result"]
        )
        assert (
            similarity_dict_only_train[k]
            == metrics_config[k]["expected_results"]["diff_result_only_train"]
        )
