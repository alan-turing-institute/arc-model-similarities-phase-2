import argparse
import logging
import os

import yaml
from utils import opts2dmpairArgs

from modsim2.data.loader import DMPair
from modsim2.model.utils import get_wandb_run

# Set logging level
logging.basicConfig(level=logging.INFO, format="%(message)s")


def main(
    experiment_group_config: dict,
    experiment_group: str,
    dmpair_config: str,
    metrics_config: dict,
    trainer_config: dict,
    dataset_index: int,
    seed_index: int,
):
    # Get experiment pair name and dmpair kwargs
    experiment_pair_name = f"{experiment_group}_{dataset_index}_{seed_index}"
    dmpair_kwargs = opts2dmpairArgs(
        opt=experiment_group_config["dmpairs"][dataset_index],
        seed=dmpair_config["seeds"][seed_index],
        val_split=dmpair_config["val_split"],
    )

    # Instantsiate DMPair, apply transforms, compute metrics
    dmpair = DMPair(**dmpair_kwargs, metrics_config=metrics_config)
    dmpair.A.setup()
    dmpair.B.setup()
    similarity_metrics = {
        **dmpair.compute_similarity(),
    }

    # Split into metrics to log to A, and metrics to log to B
    A_metrics = {key: similarity_metrics[key][0] for key in similarity_metrics}
    B_metrics = {key: similarity_metrics[key][1] for key in similarity_metrics}

    # Log to wandb - A metrics
    run_A = get_wandb_run(
        model_suffix="A",
        experiment_pair_name=experiment_pair_name,
        entity=trainer_config["wandb"]["entity"],
        project_name=trainer_config["wandb"]["project"],
    )
    run_A.log(A_metrics, commit=True)
    run_A.finish()

    # Log to wandb - B metrics
    run_B = get_wandb_run(
        model_suffix="B",
        experiment_pair_name=experiment_pair_name,
        entity=trainer_config["wandb"]["entity"],
        project_name=trainer_config["wandb"]["project"],
    )
    run_B.log(B_metrics, commit=True)
    run_B.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        This script takes some paths to dataset and metrics configs, along with
        arguments to the dataset config as inputs. The arguments to the dataset
        config specify which dataset pair to perform the experiment on.

        It then calculates the dataset similarity metrics specifeid in the metrics
        config, and logs the results to wandb. See README for more information.
        """
    )
    parser.add_argument(
        "--experiment_groups_path",
        type=str,
        help="path to experiment groups config folder",
        required=True,
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
        required=True,
    )
    parser.add_argument(
        "--metrics_config_path",
        type=str,
        help="path to metrics config file",
        required=True,
    )
    parser.add_argument(
        "--metrics_file",
        type=str,
        help="metrics yaml file to use",
        required=True,
    )
    parser.add_argument(
        "--trainer_config_path",
        type=str,
        help="path to trainer config file",
        required=True,
    )
    parser.add_argument(
        "--dataset_index",
        type=int,
        help="index of dataset options within group",
        required=True,
    )
    parser.add_argument(
        "--seed_index", type=int, help="index of seed within seeds", required=True
    )

    args = parser.parse_args()

    with open(
        os.path.join(args.experiment_groups_path, args.experiment_group + ".yaml"), "r"
    ) as stream:
        experiment_group_config = yaml.safe_load(stream)

    with open(args.dmpair_config_path, "r") as stream:
        dmpair_config = yaml.safe_load(stream)

    with open(
        os.path.join(args.metrics_config_path, args.metrics_file + ".yaml"), "r"
    ) as stream:
        metrics_config = yaml.safe_load(stream)

    with open(args.trainer_config_path, "r") as stream:
        trainer_config = yaml.safe_load(stream)

    main(
        experiment_group_config=experiment_group_config,
        experiment_group=args.experiment_group,
        dmpair_config=dmpair_config,
        metrics_config=metrics_config["metrics"],
        trainer_config=trainer_config,
        dataset_index=args.dataset_index,
        seed_index=args.seed_index,
    )
