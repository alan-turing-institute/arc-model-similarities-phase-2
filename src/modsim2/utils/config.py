from typing import Optional

import torch.nn
import torchvision.transforms
import yaml


def load_configs(
    metrics_config_path: Optional[str], transforms_config_path: Optional[str] = None
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

    if metrics_config_path:
        with open(metrics_config_path, "r") as stream:
            config_dict["metrics_config"] = yaml.safe_load(stream)

    if transforms_config_path:
        with open(transforms_config_path, "r") as stream:
            config_dict["transforms_config"] = yaml.safe_load(stream)

    # Attach to single output dict
    return config_dict


def compose_transform(
    t_conf: dict, scale_data: bool = True
) -> torchvision.transforms.transforms.Compose:
    """
    A function that takes a transform config as input, and returns a composed
    transform that turns the PIL image back into a tensor as outpt. Only allows
    for one transform to be used. Allows None to be supplied in lieu of a config,
    and returns None in this case.

    Args:
        t_conf: A dict specifying the transform, with "name" and "kwargs" keys.
        scale_data: Whether ToTensor (which scales the tensors to [0,1]) or
                    PILToTensor (which leaves the data unscaled) should be used.
                    If True, uses ToTensor.

    Returns:
        torchvision.transforms.transforms.Compose: A composed transform
    """
    # If None is supplied, return None
    if t_conf is None:
        return t_conf

    # Otherwise, load the transform
    transform = _load_transform(t_conf)

    # Transforms create PIL images. We need to return tensors though
    if scale_data:
        to_tensor = torchvision.transforms.ToTensor()
    else:
        to_tensor = torchvision.transforms.PILToTensor()

    # Compose and return the transform
    composed = torchvision.transforms.Compose([transform, to_tensor])
    return composed


def _load_transform(t_conf: dict) -> torch.nn.Module:
    clazz = getattr(torchvision.transforms, t_conf["name"])
    kwargs = t_conf.get("kwargs", {})  # default to empty
    return clazz(**kwargs)


def create_transforms(transforms_list: list[dict]) -> torch.nn.Module:
    transforms = [_load_transform(t_conf=t_conf) for t_conf in transforms_list]
    return torchvision.transforms.Compose(transforms)
