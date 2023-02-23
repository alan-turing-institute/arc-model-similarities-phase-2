from typing import Optional

import torch.nn
import torchvision.transforms
import yaml


def load_configs(
    metrics_config_path: str, transforms_config_path: Optional[str] = None
) -> dict:
    """
    Function that takes config paths as input and returns a dictionary
    for configuring experiments as output

    Args:
        metrics_config_path: path to YAML config file for dataset similarity metrics
        transforms_config_path: path to YAML config file for torchvision transformations

    Returns:
        Dict: dictionary of configs
    """

    # Outputs inside a single, overarching config
    config_dict = {}

    with open(metrics_config_path, "r") as stream:
        config_dict["metric_config"] = yaml.safe_load(stream)

    if transforms_config_path:
        with open(transforms_config_path, "r") as stream:
            config_dict["transforms_config"] = yaml.safe_load(stream)

    # Attach to single output dict
    return config_dict


def _load_transform(t_conf: dict) -> torch.nn.Module:
    clazz = getattr(torchvision.transforms, t_conf["name"])
    kwargs = t_conf.get("kwargs", {})  # default to empty
    return clazz(**kwargs)


def create_transforms(transforms_list: list[dict]) -> torch.nn.Module:
    transforms = [_load_transform(t_conf=t_conf) for t_conf in transforms_list]
    return torchvision.transforms.Compose(transforms)
