import os

from modsim2.data.loader import DMPair
from modsim2.utils.config import load_configs

# project root = arc-model-similarites-phase-2/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Get metric config
METRICS_CONFIG_PATH = os.path.join(PROJECT_ROOT, "tests", "testconfig", "metrics.yaml")
CONFIGS = load_configs(METRICS_CONFIG_PATH)
METRIC_CONFIG = CONFIGS["metric_config"]
METRIC_CONFIG = {k: v for k, v in METRIC_CONFIG.items() if v["function"] == "ot"}


def test_cifar_ot_same():
    dmpair = DMPair(metric_config=METRIC_CONFIG)

    similarity_dict = dmpair.compute_similarity(only_train=True, return_dataset=True)
    for k in similarity_dict:
        print(k + " similarity distance:", float(similarity_dict[k]))
        assert float(similarity_dict[k]) == 0
