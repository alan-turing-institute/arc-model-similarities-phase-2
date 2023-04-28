import os
import platform

import pytest

from modsim2.data.loader import DMPair
from modsim2.utils.config import load_configs


# Fixture that returns the metric config dictionary
@pytest.fixture(scope="module")
def metrics_config() -> dict:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    metrics_config_path = os.path.join(
        project_root, "tests", "testconfig", "metrics.yaml"
    )
    otdd_config = load_configs(metrics_config_path)["metrics_config"]
    # filter down to only otdd configs
    otdd_config = {k: v for k, v in otdd_config.items() if v["class"] == "otdd"}

    return otdd_config


# This test checks that the distance between a dataset and itself is the expect value
# When exact calculations are used then this will be zero
# Approximate methods may be non-zero and these are checked against the known value
# for the seed
@pytest.mark.skipif(
    platform.processor() == "arm",
    reason="These tests should not be run on Apple M1 devices",
)
def test_cifar_otdd_same(metrics_config: dict):
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
@pytest.mark.skipif(
    platform.processor() == "arm",
    reason="These tests should not be run on Apple M1 devices",
)
def test_cifar_otdd_different(metrics_config: dict):
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


# This test checks that a value error is raised wen
@pytest.mark.skipif(
    platform.processor() != "arm",
    reason="This test is only applicable to Apple M1 devices",
)
def test_cifar_otdd_raise_error(metrics_config: dict):
    dmpair = DMPair(metrics_config=metrics_config)
    with pytest.raises(ValueError):
        dmpair.compute_similarity(only_train=False)
