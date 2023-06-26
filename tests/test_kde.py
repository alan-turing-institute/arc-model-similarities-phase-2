import os

import pytest
import yaml
from testing_utils import assert_all_scenarios_0, assert_all_scenarios_above_0

from modsim2.data.loader import DMPair


# Fixture that returns the metric config dictionary
@pytest.fixture(scope="module")
def metrics_config() -> dict:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    metrics_config_path = os.path.join(
        project_root, "tests", "testconfig", "metrics.yaml"
    )
    with open(metrics_config_path, "r") as stream:
        kde_config = yaml.safe_load(stream)["metrics"]
    # filter down to only pad configs
    kde_config = {k: v for k, v in kde_config.items() if v["class"] == "kde"}
    return kde_config


# This test checks that the distance between a dataset and itself returns zero
def test_cifar_kde_same(
    metrics_config: dict, fixedInceptionMock, fixedUmapMock, fixedPcaMock
):
    dmpair = DMPair(metrics_config=metrics_config, seed=42)
    similarity_dict = dmpair.compute_similarity(only_train=False)
    similarity_dict_only_train = dmpair.compute_similarity(only_train=True)
    test_scenarios = {
        "same_result": similarity_dict,
        "same_result_only_train": similarity_dict_only_train,
    }
    assert_all_scenarios_0(test_scenarios=test_scenarios)


# This test checks that the distance between two different datasets is the expected
# value
# The calculated distance is checked against the known value for the seed - brittle
# test but useful for messing with code
def test_cifar_kde_different(
    metrics_config: dict,
    randomNormalDifferentInceptionMock,
    randomNormalDifferentUmapMock,
    randomNormalDifferentPcaMock,
):

    dmpair = DMPair(
        metrics_config=metrics_config, drop_percent_A=0.5, drop_percent_B=0.5, seed=42
    )

    similarity_dict = dmpair.compute_similarity()
    similarity_dict_only_train = dmpair.compute_similarity(only_train=True)

    test_scenarios = {
        "diff_result": similarity_dict,
        "diff_result_only_train": similarity_dict_only_train,
    }

    assert_all_scenarios_above_0(test_scenarios=test_scenarios)
