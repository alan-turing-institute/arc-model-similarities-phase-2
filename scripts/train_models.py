import argparse

import yaml
from pytorch_lightning import LightningDataModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger

import wandb
from modsim2.data.loader import DMPair
from modsim2.model.resnet import ResnetModel


def train_model(dm: LightningDataModule, experiment_name: str, trainer_config: dict):
    model = ResnetModel(**trainer_config["model"])

    wandb_logger = WandbLogger(
        entity=trainer_config["wandb"]["entity"],
        project=trainer_config["wandb"]["project"],
        name=experiment_name,
        mode=trainer_config["wandb"]["mode"],
    )

    trainer = Trainer(
        **trainer_config["trainer_kwargs"],
        logger=wandb_logger,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            TQDMProgressBar(refresh_rate=10),
        ],
    )

    trainer.fit(model, dm)
    # manually finish after each run
    wandb.finish()


def train_models(dmpair_kwargs: dict, trainer_config: dict, experiment_pair_name: str):
    dmpair = DMPair(**dmpair_kwargs)
    train_model(
        dm=dmpair.A,
        experiment_name=f"{experiment_pair_name}_A",
        trainer_config=trainer_config,
    )
    train_model(
        dm=dmpair.B,
        experiment_name=f"{experiment_pair_name}_B",
        trainer_config=trainer_config,
    )


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
        "--dataset_config", type=str, help="path to datasets config file"
    )
    parser.add_argument(
        "--experiment_group", type=str, help="which experiment group to run"
    )
    parser.add_argument(
        "--trainer_config", type=str, help="path to trainer config file"
    )
    parser.add_argument(
        "--dataset_index", type=int, help="index of dataset options within group"
    )
    parser.add_argument("--seed_index", type=int, help="index of seed within seeds")

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
