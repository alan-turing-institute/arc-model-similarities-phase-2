import os
import platform

import numpy as np
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


# This test checks that the distance between a dataset and itself is close to the
# expected value within a tolerance
# A check for equality is not performed as different processors can return slightly
# different values
# Approximate methods may be non-zero when comparing a dataset to itself and these
# are checked against the known value for the seed
@pytest.mark.skipif(
    platform.processor() == "arm",
    reason="These tests should not be run on Apple M1 devices",
)
def test_cifar_otdd_same(metrics_config: dict):
    dmpair = DMPair(metrics_config=metrics_config, seed=42)
    similarity_dict = dmpair.compute_similarity(only_train=False)
    similarity_dict_only_train = dmpair.compute_similarity(only_train=True)
    failures = []
    for results in [
        ("same_result", similarity_dict),
        ("same_result_only_train", similarity_dict_only_train),
    ]:
        for k in metrics_config:
            test_name = k + "/" + results[0]
            expected_result = metrics_config[k]["expected_results"]["same_result"]
            actual_result = results[1][k]
            if not np.isclose(actual_result, expected_result, rtol=1e-5, atol=1e-8):
                failure = (
                    "test:"
                    + test_name
                    + ", expected result: "
                    + str(expected_result)
                    + ", actual result: "
                    + str(actual_result)
                )
                failures.append(failure)
    if failures:
        pytest.fail("\n".join(failures))


# This test checks that the distance between two different datasets is close to the
# expected value within a tolerance
# A check for equality is not performed as different processors can return slightly
# different values
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
    failures = []
    for results in [
        ("same_result", similarity_dict),
        ("same_result_only_train", similarity_dict_only_train),
    ]:
        for k in metrics_config:
            test_name = k + "/" + results[0]
            expected_result = metrics_config[k]["expected_results"]["same_result"]
            actual_result = results[1][k]
            if not np.isclose(actual_result, expected_result, rtol=1e-5, atol=1e-8):
                failure = (
                    "test:"
                    + test_name
                    + ", expected result: "
                    + str(expected_result)
                    + ", actual result: "
                    + str(actual_result)
                )
                failures.append(failure)
    if failures:
        pytest.fail("\n".join(failures))


# This test checks that a value error is raised wen
@pytest.mark.skipif(
    platform.processor() != "arm",
    reason="This test is only applicable to Apple M1 devices",
)
def test_cifar_otdd_raise_error(metrics_config: dict):
    dmpair = DMPair(metrics_config=metrics_config)
    with pytest.raises(ValueError):
        dmpair.compute_similarity(only_train=False)
