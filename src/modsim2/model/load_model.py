import os

import torch

import wandb
from modsim2.model.resnet import ResnetModel


def download_model(
    experiment_name: str,
    entity: str,
    project_name: str,
    id_postfix: str,
    version: str,
) -> tuple(ResnetModel, wandb.run):
    """
    A function that restores a wandb run and downloads the corresponding model artifact,
    then loads it as a ResnetModel.

    Args:
        experiment_name: Name of the experiment (e.g. drop-only_1_0_A)
        entity: Wandb entity name
        project_name: Wandb project name
        id_postfix: Postfix added to experiment_name for the unique ID
        version: Model version to use. Recommended :latest

    Returns: The ResnetModel and wandb run
    """
    # Names to use in resuming the run and downloading the model
    model_name = experiment_name + "_model"
    folder = entity + "/" + project_name + "/"
    experiment_id = experiment_name + id_postfix

    # Download the model
    run = wandb.init(
        project=project_name,
        entity=entity,
        name=experiment_name,
        id=experiment_id,
        resume=True,
    )
    artifact = run.use_artifact(folder + model_name + version, type="model")
    artifact_dir = artifact.download()

    # Read in the model
    path = os.path.join(artifact_dir, "model.ckpt")
    checkpoint = torch.load(path, map_location=torch.device("cpu"))
    model = ResnetModel()
    model.load_state_dict(checkpoint["state_dict"])

    return model, run
