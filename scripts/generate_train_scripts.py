import argparse
import os

import constants
import utils
import yaml


def main(
    experiment_groups_path,
    experiment_group,
    dmpair_config_path,
    trainer_config_path,
    account_name,
    conda_env_path,
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
        + f"--dmpair_config_path {dmpair_config_path} "
        + f"--trainer_config_path {trainer_config_path} "
        + f"--seed_index {seed_index} "
        + f"--dataset_index {dataset_index}"
        for seed_index in range(NUM_SEEDS)
        for dataset_index in range(NUM_PAIRS)
    ]

    # Prepare path
    scripts_path = os.path.join(constants.PROJECT_ROOT, "train_scripts")

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
        os.path.join(scripts_path, experiment_name + "_train.sh")
        for experiment_name in experiment_names
    ]

    utils.write_slurm_script(
        template_name="slurm-train-template.sh",
        called_script_name="train_models.py",
        combinations=combinations,
        account_name=account_name,
        experiment_names=experiment_names,
        conda_env_path=conda_env_path,
        generated_script_names=script_names,
    )


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
        "--trainer_config_path",
        type=str,
        help="path to train config file",
        default=constants.TRAINER_CONFIG_PATH,
    )
    parser.add_argument(
        "--account_name", type=str, help="", default="vjgo8416-mod-sim-2"
    )
    parser.add_argument(
        "--conda_env_path",
        type=str,
        help="path to conda env",
        default="/bask/projects/v/vjgo8416-mod-sim-2/ms2env",
    )
    args = parser.parse_args()
    main(
        experiment_groups_path=args.experiment_groups_path,
        experiment_group=args.experiment_group,
        dmpair_config_path=args.dmpair_config_path,
        trainer_config_path=args.trainer_config_path,
        account_name=args.account_name,
        conda_env_path=args.conda_env_path,
    )
