import os
from unittest.mock import patch

import numpy as np
import pytest
import yaml
from pytest_check import check

from modsim2.data.loader import DMPair
from modsim2.similarity.metrics import pad


# Fixture that returns the metric config dictionary
@pytest.fixture(scope="module")
def metrics_config() -> dict:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    metrics_config_path = os.path.join(
        project_root, "tests", "testconfig", "metrics.yaml"
    )
    with open(metrics_config_path, "r") as stream:
        pad_config = yaml.safe_load(stream)
    # filter down to only pad configs
    pad_config = {k: v for k, v in pad_config.items() if v["class"] == "pad"}
    return pad_config


# This test checks that the distance between a dataset and itself is the expect value
# For this metric the expected value is always zero
def test_cifar_pad_same(metrics_config: dict):
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
def test_cifar_pad_different(metrics_config: dict):
    dmpair = DMPair(metrics_config=metrics_config, drop_percent_A=0.2, seed=42)
    similarity_dict = dmpair.compute_similarity()
    similarity_dict_only_train = dmpair.compute_similarity(only_train=True)
    test_scenarios = {
        "diff_result": similarity_dict,
        "diff_result_only_train": similarity_dict_only_train,
    }
    compare_results(test_scenarios, metrics_config)


# This test checks that there are equal samples of A and B in the
# combined test dataset
def test_cifar_pad_equal_samples(metrics_config: dict):
    # Drop data from B to ensure initial sizes are not equal
    dmpair = DMPair(drop_percent_B=0.5, metrics_config=metrics_config)
    train_data_A, val_data_A = dmpair.get_A_data()
    train_data_B, val_data_B = dmpair.get_B_data()

    train_labels_A, val_labels_A = dmpair.get_A_labels()
    train_labels_B, val_labels_B = dmpair.get_B_labels()

    data_A = np.concatenate((train_data_A, val_data_A), axis=0)
    data_B = np.concatenate((train_data_B, val_data_B), axis=0)

    labels_A = np.concatenate((train_labels_A, val_labels_A), axis=0)
    labels_B = np.concatenate((train_labels_B, val_labels_B), axis=0)
    for _, metric in metrics_config.items():

        pad_metric = pad.PAD(seed=dmpair.seed)
        with patch.object(
            pad_metric, "_evaluate_models", wraps=pad_metric._evaluate_models
        ) as wrapped_evaluate_models:

            _ = pad_metric.calculate_distance(
                data_A=data_A,
                data_B=data_B,
                labels_A=labels_A,
                labels_B=labels_B,
                **metric["arguments"],
            )
            combined_test_labels = wrapped_evaluate_models.call_args.kwargs[
                "test_labels"
            ]
            sum_test_labels = np.sum(combined_test_labels)
            count_test_labels = combined_test_labels.shape[0]

        # The label values are either zero or one, so the sum of the labels should
        # equal half the number of records if the dataset has been balanced, or
        # 1.5 if not (due to dropping 50% of B)
        if metric["arguments"]["balance_test"]:
            with check:
                assert (2 * sum_test_labels) == count_test_labels
        if not metric["arguments"]["balance_test"]:
            with check:
                assert (1.5 * sum_test_labels) == count_test_labels


# This function takes the computed distances (stored in test_scenarios) and
# compares them to the expected distances (stored in metrics_config)
def compare_results(test_scenarios: dict, metrics_config: dict):
    for scenario, results in test_scenarios.items():
        for k in metrics_config:
            expected_result = metrics_config[k]["expected_results"][scenario]
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
