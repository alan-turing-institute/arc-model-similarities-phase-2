import os

import torch

from modsim2.model.resnet import ResnetModel
from modsim2.model.utils import get_wandb_run


def _download_model(
    model_suffix: str,
    experiment_pair_name: str,
    entity: str,
    project_name: str,
    version: str,
) -> ResnetModel:
    """
    A function that restores a wandb run and downloads the corresponding model artifact,
    then loads it as a ResnetModel. It then closes the run and returns the model

    Args:
        model_suffix: Suffix for the model to be downloaded (A or B)
        experiment_pair_name: Name of the experiment (e.g. drop-only_1_0)
        entity: Wandb entity name
        project_name: Wandb project name
        id_postfix: Postfix added to experiment_pair_name for the unique ID
        version: Model version to use. Recommended :latest

    Returns: The ResnetModel
    """
    # Names to use in resuming the run and downloading the model
    model_name = experiment_pair_name + "_" + model_suffix + "_model"
    folder = os.path.join(entity, project_name)

    # Reinitialise the run
    run = get_wandb_run(
        model_suffix=model_suffix,
        experiment_pair_name=experiment_pair_name,
        entity=entity,
        project_name=project_name,
    )

    # Download the model
    artifact = run.use_artifact(
        os.path.join(folder, model_name + version), type="model"
    )
    artifact_dir = artifact.download()

    # Read in the model
    path = os.path.join(artifact_dir, "model.ckpt")
    checkpoint = torch.load(path, map_location=torch.device("cpu"))
    model = ResnetModel()
    model.load_state_dict(checkpoint["state_dict"])

    # Close the run
    run.finish()

    # Returb the model
    return model


def download_AB_models(
    experiment_pair_name: str,
    entity: str,
    project_name: str,
    version: str,
) -> tuple[ResnetModel, ResnetModel]:
    """
    A function that takes as input a description of a set of two models corresponding
    to an experiment pair on wandb, and returns both corresponding model artifacts as
    ResnetModels.

    Args:
        experiment_pair_name: Name of the experiment (e.g. drop-only_1_0)
        entity: Wandb entity name
        project_name: Wandb project name
        id_postfix: Postfix added to experiment_pair_name for the unique ID
        version: Model version to use. Recommended :latest

    Returns: A tuple of both ResnetModels
    """
    model_A = _download_model(
        model_suffix="A",
        experiment_pair_name=experiment_pair_name,
        entity=entity,
        project_name=project_name,
        version=version,
    )
    model_B = _download_model(
        model_suffix="B",
        experiment_pair_name=experiment_pair_name,
        entity=entity,
        project_name=project_name,
        version=version,
    )
    return model_A, model_B
