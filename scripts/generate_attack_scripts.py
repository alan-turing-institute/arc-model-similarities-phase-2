import argparse
import os

import constants
import yaml


def main(
    experiment_group,
    dataset_config_path,
    trainer_config_path,
    attack_config_path,
):
    # Dataset config
    with open(dataset_config_path, "r") as stream:
        dataset_config = yaml.safe_load(stream)

    # Prepare dataset list, seed list
    NUM_SEEDS = len(dataset_config["seeds"])
    NUM_PAIRS = len(dataset_config["experiment_groups"][experiment_group])

    # Generate combinations of arguments to pass
    combinations = [
        f"--dataset_config {dataset_config_path} "
        + f"--trainer_config {trainer_config_path} "
        + f"--attack_config {attack_config_path}"
        + f"--experiment_group {experiment_group} "
        + f"--dataset_index {dataset_index}"
        + f"--seed_index {seed_index} "
        for seed_index in range(NUM_SEEDS)
        for dataset_index in range(NUM_PAIRS)
    ]

    # Prepare path
    scripts_path = os.path.join(constants.PROJECT_ROOT, "attack_scripts")

    # If bash scripts path does not exist, create it
    if not os.path.isdir(scripts_path):
        os.mkdir(scripts_path)

    # Generate files + script names
    experiment_names = [
        f"{experiment_group}_{dataset_index}_{seed_index}"
        for seed_index in range(NUM_SEEDS)
        for dataset_index in range(NUM_PAIRS)
    ]
    script_names = [
        scripts_path + "/" + experiment_name + "_attack.sh"
        for experiment_name in experiment_names
    ]

    # For each combination, write bash script with params
    for index, combo in enumerate(combinations):
        python_call = f"python scripts/attack.py {combo}"
        with open(script_names[index], "w") as f:
            f.write(python_call)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process arguments for script generation."
    )
    parser.add_argument(
        "--experiment_group",
        type=str,
        help="experiment group to use",
        required=True,
    )
    parser.add_argument(
        "--dataset_config_path",
        type=str,
        help="path to datasets config file",
        default=constants.DATASET_CONFIG_PATH,
    )
    parser.add_argument(
        "--trainer_config_path",
        type=str,
        help="path to train config file",
        default=constants.TRAINER_CONFIG_PATH,
    )
    parser.add_argument(
        "--attack_config_path",
        type=str,
        help="path to attack config file",
        default=constants.ATTACK_CONFIG_PATH,
    )
    args = parser.parse_args()
    main(
        experiment_group=args.experiment_group,
        dataset_config_path=args.dataset_config_path,
        trainer_config_path=args.trainer_config_path,
        attack_config_path=args.attack_config_path,
    )
