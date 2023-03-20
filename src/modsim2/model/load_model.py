import os
from typing import Optional

import torch

import wandb
from modsim2.model.resnet import ResnetModel


def download_model(
    experiment_name: str,
    entity="turing-arc",
    project_name="ms2",
    version: Optional[str] = ":latest",  # TODO: argparse in script
) -> ResnetModel:
    # Names to use in restoring the model
    model_name = experiment_name + "_model"
    folder = entity + "/" + project_name + "/"
    experiment_id = experiment_name + "_test"  # TODO: change this to a config

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
    # run.finish()

    # Read in the model
    path = os.path.join(artifact_dir, "model.ckpt")
    checkpoint = torch.load(path, map_location=torch.device("cpu"))
    model = ResnetModel()
    model.load_state_dict(checkpoint["state_dict"])

    return model, run
