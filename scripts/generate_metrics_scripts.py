import argparse
import os

import constants
import yaml

# Selections


def main(
    experiment_groups_path, experiment_group, dmpair_config_path, metrics_config_path
):
    # Dataset config
    with open(
        os.path.join(experiment_groups_path, experiment_group + ".yaml")
    ) as stream:
        experiment_group_config = yaml.safe_load(stream)

    with open(dmpair_config_path, "r") as stream:
        dmpair_config = yaml.safe_load(stream)

    # Prepare dataset list, seed list
    NUM_SEEDS = len(dmpair_config["seeds"])
    NUM_PAIRS = len(experiment_group_config["dmpairs"])

    # Generate combinations of arguments to pass
    combinations = [
        f"--experiment_groups_path {experiment_groups_path} "
        + f"--experiment_group {experiment_group} "
        + f"--dmpair_config {dmpair_config_path} "
        + f"--metrics_config {metrics_config_path} "
        + f"--seed_index {seed_index} "
        + f"--dataset_index {dataset_index}"
        for seed_index in range(NUM_SEEDS)
        for dataset_index in range(NUM_PAIRS)
    ]

    # Prepare path
    scripts_path = os.path.join(constants.PROJECT_ROOT, "metrics_scripts")

    # If bash scripts path does not exist, create it
    if not os.path.isdir(scripts_path):
        os.mkdir(scripts_path)

    # Generate files + script names
    experiment_pair_names = [
        scripts_path
        + f"/{experiment_group}_{dataset_index}_{seed_index}"
        + "_metrics.sh"
        for seed_index in range(NUM_SEEDS)
        for dataset_index in range(NUM_PAIRS)
    ]

    # For each combination, write bash script with params
    # TODO: replace script writing with slurm
    for index, combo in enumerate(combinations):
        script = f"python scripts/calculate_metrics.py {combo}"
        with open(experiment_pair_names[index], "w") as f:
            f.write(script)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process arguments for script generation."
    )
    parser.add_argument(
        "--experiment_groups_path",
        type=str,
        help="path to experiment groups config folder",
        default=constants.EXPERIMENT_GROUPS_PATH,
    )
    parser.add_argument(
        "--experiment_group",
        type=str,
        help="experiment group to use",
        required=True,
    )
    parser.add_argument(
        "--dmpair_config_path",
        type=str,
        help="path to dmpair config file",
        default=constants.DMPAIR_CONFIG_PATH,
    )
    parser.add_argument(
        "--metrics_config_path",
        type=str,
        help="path to metrics config file",
        default=constants.METRICS_CONFIG_PATH,
    )
    args = parser.parse_args()
    main(
        experiment_groups_path=args.experiment_groups_path,
        experiment_group=args.experiment_group,
        dmpair_config_path=args.dmpair_config_path,
        metrics_config_path=args.metrics_config_path,
    )
