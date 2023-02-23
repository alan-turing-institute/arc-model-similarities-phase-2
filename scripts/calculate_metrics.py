import argparse
import json
import os

import yaml
from functions import opts2dmpairArgs

from modsim2.data.loader import DMPair


def main(
    dataset_config: dict,
    metric_config: dict,
    experiment_group: str,
    dataset_index: int,
    seed_index: int,
):
    experiment_pair_name = f"{experiment_group}_{dataset_index}_{seed_index}"
    dmpair_kwargs = opts2dmpairArgs(
        opt=dataset_config["experiment_groups"][experiment_group][dataset_index],
        seed=dataset_config["seeds"][seed_index],
    )

    dmpair = DMPair(**dmpair_kwargs, metric_config=metric_config)
    metrics = {
        "experiment_pair_name": experiment_pair_name,
        **dmpair.compute_similarity(),
    }

    # JSON file path
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    data_path = os.path.join(root_path, "data")
    out_file_path = os.path.join(data_path, "metrics.json")

    # Create data folder if it does not exist
    if not os.path.isdir(data_path):
        os.mkdir(data_path)

    # If file does not exist, create it
    file_exists = os.path.isfile(out_file_path)
    if not file_exists:
        with open(out_file_path, "w") as out_file:
            json.dump([metrics], out_file, indent=4)

    # If file exists, append to it
    if file_exists:
        # Read file contents
        with open(out_file_path) as out_file:
            out_file_contents = json.load(out_file)

        # Append dict to list in file
        out_file_contents.append(metrics)

        # Write new list with new output to file
        with open(out_file_path, "w") as out_file:
            json.dump(out_file_contents, out_file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--dataset_config", type=str, help="path to datasets config file"
    )
    parser.add_argument(
        "--experiment_group", type=str, help="which experiment group to run"
    )
    parser.add_argument("--metric_config", type=str, help="path to metric config file")
    parser.add_argument(
        "--dataset_index", type=int, help="index of dataset options within group"
    )
    parser.add_argument("--seed_index", type=int, help="index of seed within seeds")

    args = parser.parse_args()

    with open(args.dataset_config, "r") as stream:
        dataset_config = yaml.safe_load(stream)

    with open(args.metric_config, "r") as stream:
        metric_config = yaml.safe_load(stream)

    main(
        dataset_config=dataset_config,
        metric_config=metric_config,
        experiment_group=args.experiment_group,
        dataset_index=args.dataset_index,
        seed_index=args.seed_index,
    )
