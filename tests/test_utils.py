import os

from modsim2.utils.config import load_configs

# project root = arc-model-similarites-phase-2/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Get configs
METRICS_CONFIG_PATH = os.path.join(PROJECT_ROOT, "tests", "testconfig", "metrics.yaml")
CONFIGS = load_configs(METRICS_CONFIG_PATH)


def test_configs_exist():
    assert "metric_config" in CONFIGS


def test_metric_configs_structure_exists():
    metric_config = CONFIGS["metric_config"]
    assert "mmd_rbf" in metric_config
    assert "function" in metric_config["mmd_rbf"]
    assert "arguments" in metric_config["mmd_rbf"]
    assert "embedding_name" in metric_config["mmd_rbf"]["arguments"]
    assert "kernel_name" in metric_config["mmd_rbf"]["arguments"]
