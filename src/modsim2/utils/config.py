import torch.nn
import torchvision.transforms


def _load_transform(t_conf: dict) -> torch.nn.Module:
    clazz = getattr(torchvision.transforms, t_conf["name"])
    kwargs = t_conf.get("kwargs", {})  # default to empty
    return clazz(**kwargs)


def create_transforms(
    transforms_list: list[dict],
) -> torchvision.transforms.transforms.Compose:
    """
    A function that takes a list of transform configs as input, and returns a composed
    transform. Allows None to be supplied in lieu of a config, and returns None in this
    case.

    Args:
        transforms_list: A list of dicts, each element of which specifies a single
                         transform with "name" and "kwargs" keys.
        scale_data: Whether ToTensor (which scales the tensors to [0,1]) or
                    PILToTensor (which leaves the data unscaled) should be used.
                    If True, uses ToTensor.

    Returns:
        torchvision.transforms.transforms.Compose: A composed transform
    """
    if transforms_list is None:
        return transforms_list
    transforms = [_load_transform(t_conf=t_conf) for t_conf in transforms_list]
    return torchvision.transforms.Compose(transforms)
