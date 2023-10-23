import os

import torchvision.transforms
import yaml

from modsim2.utils.config import create_transforms

# project root = arc-model-similarites-phase-2/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Get configs
METRICS_CONFIG_PATH = os.path.join(PROJECT_ROOT, "tests", "testconfig", "metrics.yaml")
TRANSFORMS_CONFIG_PATH = os.path.join(
    PROJECT_ROOT, "tests", "testconfig", "transforms.yaml"
)
with open(METRICS_CONFIG_PATH, "r") as stream:
    METRICS_CONFIG = yaml.safe_load(stream)["metrics"]
with open(TRANSFORMS_CONFIG_PATH, "r") as stream:
    TRANSFORMS_CONFIG_PATH = yaml.safe_load(stream)


def test_metric_configs_structure_exists():
    assert "mmd_rbf" in METRICS_CONFIG
    assert "class" in METRICS_CONFIG["mmd_rbf"]
    assert "arguments" in METRICS_CONFIG["mmd_rbf"]
    assert "embedding_name" in METRICS_CONFIG["mmd_rbf"]["arguments"]
    assert "kernel_name" in METRICS_CONFIG["mmd_rbf"]["arguments"]


def test_load_transform():
    transforms_config = TRANSFORMS_CONFIG_PATH["dmpairs"]
    transforms = create_transforms(
        transforms_list=transforms_config[0]["A"]["transforms"]
    )
    assert type(transforms.transforms[0]) == torchvision.transforms.ToTensor
    assert type(transforms.transforms[1]) == torchvision.transforms.Normalize
    assert transforms.transforms[1].mean == [0.2, 0.3, 0.4]
    assert transforms.transforms[1].std == [0.1, 0.2, 0.3]
