import argparse
import os

import yaml
from pytorch_lightning import LightningDataModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.utilities.seed import seed_everything
from utils import opts2dmpairArgs

import wandb
from modsim2.data.loader import DMPair
from modsim2.model.resnet import ResnetModel
from modsim2.model.utils import run_exists
from modsim2.model.wandb_logger import MS2WandbLogger


def train_model(dm: LightningDataModule, experiment_name: str, trainer_config: dict):
    if run_exists(
        run_name=experiment_name,
        entity=trainer_config["wandb"]["entity"],
        project_name=trainer_config["wandb"]["project"],
    ):
        raise Exception(
            "A run with this name already exists on wandb. Please rename or delete it."
        )

    model = ResnetModel(**trainer_config["model"], train_size=len(dm.dataset_train))

    wandb_logger = MS2WandbLogger(
        entity=trainer_config["wandb"]["entity"],
        project=trainer_config["wandb"]["project"],
        name=experiment_name,
        mode=trainer_config["wandb"]["mode"],
        log_model=True,
        checkpoint_name=experiment_name + "_model",
    )

    trainer = Trainer(
        **trainer_config["trainer_kwargs"],
        logger=wandb_logger,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            TQDMProgressBar(refresh_rate=10),
        ],
        deterministic=True,
    )

    trainer.fit(model, dm)
    # manually finish after each run
    wandb.finish()


def train_models(dmpair_kwargs: dict, trainer_config: dict, experiment_pair_name: str):
    dmpair = DMPair(**dmpair_kwargs)
    seed_everything(seed=dmpair_kwargs["seed"])
    train_model(
        dm=dmpair.A,
        experiment_name=f"{experiment_pair_name}_A",
        trainer_config=trainer_config,
    )
    seed_everything(seed=dmpair_kwargs["seed"])
    train_model(
        dm=dmpair.B,
        experiment_name=f"{experiment_pair_name}_B",
        trainer_config=trainer_config,
    )


def main(
    experiment_group_config: dict,
    experiment_group: str,
    dmpair_config: str,
    trainer_config: dict,
    dataset_index: int,
    seed_index: int,
):
    experiment_pair_name = f"{experiment_group}_{dataset_index}_{seed_index}"
    dmpair_kwargs = opts2dmpairArgs(
        opt=experiment_group_config["dmpairs"][dataset_index],
        seed=dmpair_config["seeds"][seed_index],
        val_split=dmpair_config["val_split"],
    )
    train_models(
        dmpair_kwargs=dmpair_kwargs,
        trainer_config=trainer_config,
        experiment_pair_name=experiment_pair_name,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        This script takes some paths to dataset, and trainer configs, along with
        arguments to the dataset config as inputs. The arguments to the dataset
        config specify which dataset pair to perform the experiment on.

        It then trains both models for the experiment with the parameters specified
        in the trainer config, and logs the results to wandb. See README for more
        information.
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
        "--seed_index",
        type=int,
        help="index of seed within seeds",
        required=True,
    )

    args = parser.parse_args()

    with open(
        os.path.join(args.experiment_groups_path, args.experiment_group + ".yaml"), "r"
    ) as stream:
        experiment_group_config = yaml.safe_load(stream)

    with open(args.dmpair_config_path, "r") as stream:
        dmpair_config = yaml.safe_load(stream)

    with open(args.trainer_config_path, "r") as stream:
        trainer_config = yaml.safe_load(stream)

    main(
        experiment_group_config=experiment_group_config,
        experiment_group=args.experiment_group,
        dmpair_config=dmpair_config,
        trainer_config=trainer_config,
        dataset_index=args.dataset_index,
        seed_index=args.seed_index,
    )
