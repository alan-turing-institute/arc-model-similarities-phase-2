from typing import Dict

import yaml


def load_configs(
    metrics_config_path: str,
) -> Dict:
    """
    Function that takes config paths as input and returns a dictionary
    for configuring experiments as output

    Args:
        metrics_config_path: path to YAML config file for dataset similarity metrics

    Returns:
        Dict: dictionary of configs
    """

    # Outputs inside a single, overarching config
    config_dict = {}

    with open(metrics_config_path, "r") as stream:
        config_dict["metric_config"] = yaml.safe_load(stream)

    # Attach to single output dict
    return config_dict
