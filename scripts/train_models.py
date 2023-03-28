import argparse

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

    model = ResnetModel(**trainer_config["model"])

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
    dataset_config: dict,
    trainer_config: dict,
    experiment_group: str,
    dataset_index: int,
    seed_index: int,
):
    experiment_pair_name = f"{experiment_group}_{dataset_index}_{seed_index}"
    dmpair_kwargs = opts2dmpairArgs(
        opt=dataset_config["experiment_groups"][experiment_group][dataset_index],
        seed=dataset_config["seeds"][seed_index],
    )
    train_models(
        dmpair_kwargs=dmpair_kwargs,
        trainer_config=trainer_config,
        experiment_pair_name=experiment_pair_name,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--dataset_config", type=str, help="path to datasets config file", required=True
    )
    parser.add_argument(
        "--experiment_group",
        type=str,
        help="which experiment group to run",
        required=True,
    )
    parser.add_argument(
        "--trainer_config", type=str, help="path to trainer config file", required=True
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

    with open(args.dataset_config, "r") as stream:
        dataset_config = yaml.safe_load(stream)

    with open(args.trainer_config, "r") as stream:
        trainer_config = yaml.safe_load(stream)

    main(
        dataset_config=dataset_config,
        trainer_config=trainer_config,
        experiment_group=args.experiment_group,
        dataset_index=args.dataset_index,
        seed_index=args.seed_index,
    )
