import os

from modsim2.data.loader import DMPair
from modsim2.utils.config import load_configs

# project root = arc-model-similarites-phase-2/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Get metric config
METRICS_CONFIG_PATH = os.path.join(PROJECT_ROOT, "tests", "testconfig", "metrics.yaml")
CONFIGS = load_configs(METRICS_CONFIG_PATH)
METRIC_CONFIG = CONFIGS["metric_config"]
METRIC_CONFIG = {k: v for k, v in METRIC_CONFIG.items() if v["class"] == "mmd"}


def test_cifar_mmd_same():
    dmpair = DMPair(metric_config=METRIC_CONFIG)

    similarity_dict = dmpair.compute_similarity()
    assert similarity_dict["mmd_rbf"] == 0
    assert similarity_dict["mmd_laplace"] == 0


def test_cifar_mmd_different():
    dmpair = DMPair(metric_config=METRIC_CONFIG, drop_percent_A=0.2, seed=42)
    similarity_dict = dmpair.compute_similarity()
    # known values for this seed - brittle test but useful for messing with code
    expected_mmd_rbf = 0.00012534993489587976
    expected_mmd_laplace = 0.0002326510493355638
    assert similarity_dict["mmd_rbf"] == expected_mmd_rbf
    assert similarity_dict["mmd_laplace"] == expected_mmd_laplace


def test_cifar_mmd_different_train_only():
    dmpair = DMPair(metric_config=METRIC_CONFIG, drop_percent_A=0.2, seed=42)
    similarity_dict = dmpair.compute_similarity(only_train=True)
    # known values for this seed - brittle test but useful for messing with code
    expected_mmd_rbf = 0.0001611794365776742
    expected_mmd_laplace = 0.00029474732083722976
    assert similarity_dict["mmd_rbf"] == expected_mmd_rbf
    assert similarity_dict["mmd_laplace"] == expected_mmd_laplace
