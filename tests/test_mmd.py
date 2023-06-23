import os

import pytest
import yaml
from pytest_check import check

from modsim2.data.loader import DMPair


# Fixture that returns the metric config dictionary
@pytest.fixture(scope="module")
def metrics_config() -> dict:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    metrics_config_path = os.path.join(
        project_root, "tests", "testconfig", "metrics.yaml"
    )
    with open(metrics_config_path, "r") as stream:
        mmd_config = yaml.safe_load(stream)["metrics"]
    # filter down to only mmd configs
    mmd_config = {k: v for k, v in mmd_config.items() if v["class"] == "mmd"}
    return mmd_config


# This test checks that the distance between a dataset and itself is the expect value
# For this metric the expected value is always zero
def test_cifar_mmd_same(metrics_config: dict):
    dmpair = DMPair(metrics_config=metrics_config)
    similarity_dict = dmpair.compute_similarity()
    similarity_dict_only_train = dmpair.compute_similarity(only_train=True)
    test_scenarios = {
        "same_result": similarity_dict,
        "same_result_only_train": similarity_dict_only_train,
    }
    compare_results(test_scenarios, metrics_config)


# This test checks that the distance between two different datasets is the expected
# value
# The calculated distance is checked against the known value for the seed - brittle
# test but useful for messing with code
def test_cifar_mmd_different(metrics_config: dict):
    dmpair = DMPair(metrics_config=metrics_config, drop_percent_A=0.2, seed=42)
    similarity_dict = dmpair.compute_similarity()
    similarity_dict_only_train = dmpair.compute_similarity(only_train=True)
    test_scenarios = {
        "diff_result": similarity_dict,
        "diff_result_only_train": similarity_dict_only_train,
    }
    compare_results(test_scenarios, metrics_config)


# This function takes the computed distances (stored in test_scenarios) and
# compares them to the expected distances (stored in metrics_config)
def compare_results(test_scenarios: dict, metrics_config: dict):
    for scenario, results in test_scenarios.items():
        for k in metrics_config:
            expected_result = tuple(metrics_config[k]["expected_results"][scenario])
            actual_result = results[k]
            with check:
                assert actual_result == expected_result, (
                    "test:"
                    + k
                    + "/"
                    + scenario
                    + ", expected result: "
                    + str(expected_result)
                    + ", actual result: "
                    + str(actual_result)
                )
