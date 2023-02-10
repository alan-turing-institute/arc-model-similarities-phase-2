import os

from modsim2.data.loader import DMPair
from modsim2.utils.config import load_configs

# project root = arc-model-similarites-phase-2/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Get metric config
METRICS_CONFIG_PATH = os.path.join(PROJECT_ROOT, "tests", "testconfig", "metrics.yaml")
CONFIGS = load_configs(METRICS_CONFIG_PATH)
METRIC_CONFIG = CONFIGS["metric_config"]


def test_cifar_mmd_same():
    dmpair = DMPair(metric_config=METRIC_CONFIG)
    similarity_dict = dmpair.compute_similarity()
    assert similarity_dict["mmd_rbf"] == 0
    assert similarity_dict["mmd_laplace"] == 0


def test_cifar_mmd_different():
    dmpair = DMPair(metric_config=METRIC_CONFIG, drop_percent_A=0.2)
    similarity_dict = dmpair.compute_similarity()
    assert similarity_dict["mmd_rbf"] > 0
    assert similarity_dict["mmd_laplace"] > 0