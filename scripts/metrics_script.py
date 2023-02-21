import os

import yaml

# project root = arc-model-similarites-phase-2/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Get configs
DATASET_CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "datasets.yaml")
METRICS_CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "metrics.yaml")

# Selections
EXPERIMENT_GROUP = "drop_only"


def main():
    # Dataset config
    with open(DATASET_CONFIG_PATH, "r") as stream:
        dataset_config = yaml.safe_load(stream)

    # Prepare dataset list, seed list
    NUM_SEEDS = len(dataset_config["seeds"])
    NUM_PAIRS = len(dataset_config["experiment_groups"][EXPERIMENT_GROUP])

    # Generate combinations of arguments to pass
    combinations = [
        f"--dataset_config {DATASET_CONFIG_PATH} "
        + f"--metric_config {METRICS_CONFIG_PATH} "
        + f"--experiment_group {EXPERIMENT_GROUP} "
        + f"--seed_index {seed_index} "
        + f"--dataset_index {dataset_index}"
        for seed_index in range(NUM_SEEDS)
        for dataset_index in range(NUM_PAIRS)
    ]

    # Run each combination sequentially
    for combo in combinations:
        os.system(f"python scripts/calculate_metrics.py {combo}")


if __name__ == "__main__":
    main()
