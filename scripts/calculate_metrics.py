import argparse

import yaml

from modsim2.data.loader import DMPair


def opts2dmpairArgs(opt: dict, seed: int) -> dict:
    return {
        "drop_percent_A": opt["A"]["drop"],
        "drop_percent_B": opt["B"]["drop"],
        "transforms_A": opt["A"]["transforms"],
        "transforms_B": opt["B"]["transforms"],
        "seed": seed,
    }


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
    print(metrics)
    # return(metrics)


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
