import constants
import yaml

# Selections
# TODO: replace with argparse
EXPERIMENT_GROUP = "drop-only"


def main():
    # Dataset config
    with open(constants.DATASET_CONFIG_PATH, "r") as stream:
        dataset_config = yaml.safe_load(stream)

    # Prepare dataset list, seed list
    NUM_SEEDS = len(dataset_config["seeds"])
    NUM_PAIRS = len(dataset_config["experiment_groups"][EXPERIMENT_GROUP])

    # Generate combinations of arguments to pass
    combinations = [
        f"--dataset_config {constants.DATASET_CONFIG_PATH} "
        + f"--trainer_config {constants.TRAINER_CONFIG_PATH} "
        + f"--experiment_group {EXPERIMENT_GROUP} "
        + f"--seed_index {seed_index} "
        + f"--dataset_index {dataset_index}"
        for seed_index in range(NUM_SEEDS)
        for dataset_index in range(NUM_PAIRS)
    ]
    experiment_pair_names = [
        f"scripts/{EXPERIMENT_GROUP}_{dataset_index}_{seed_index}" + "_train.sh"
        for seed_index in range(NUM_SEEDS)
        for dataset_index in range(NUM_PAIRS)
    ]

    # For each combination, write bash script with params
    # TODO: replace script writing with slurm
    for index, combo in enumerate(combinations):
        script = f"python scripts/train_models.py {combo}"
        with open(experiment_pair_names[index], "w") as f:
            f.write(script)


if __name__ == "__main__":
    main()
