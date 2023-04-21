import os

import torchvision.transforms

from modsim2.utils.config import create_transforms, load_configs

# project root = arc-model-similarites-phase-2/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Get configs
METRICS_CONFIG_PATH = os.path.join(PROJECT_ROOT, "tests", "testconfig", "metrics.yaml")
TRANSFORMS_CONFIG_PATH = os.path.join(
    PROJECT_ROOT, "tests", "testconfig", "transforms.yaml"
)
CONFIGS = load_configs(
    metrics_config_path=METRICS_CONFIG_PATH,
    transforms_config_path=TRANSFORMS_CONFIG_PATH,
)


def test_configs_exist():
    assert "metrics_config" in CONFIGS
    assert "transforms_config" in CONFIGS


def test_metric_configs_structure_exists():
    metrics_config = CONFIGS["metrics_config"]
    assert "mmd_rbf" in metrics_config
    assert "class" in metrics_config["mmd_rbf"]
    assert "arguments" in metrics_config["mmd_rbf"]
    assert "embedding_name" in metrics_config["mmd_rbf"]["arguments"]
    assert "kernel_name" in metrics_config["mmd_rbf"]["arguments"]


def test_load_transform():
    transforms_config = CONFIGS["transforms_config"]
    transforms = create_transforms(transforms_list=transforms_config["A"])
    assert type(transforms.transforms[0]) == torchvision.transforms.ToTensor
    assert type(transforms.transforms[1]) == torchvision.transforms.Normalize
    assert transforms.transforms[1].mean == [0.2, 0.3, 0.4]
    assert transforms.transforms[1].std == [0.1, 0.2, 0.3]
