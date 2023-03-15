import os
from typing import Optional

import torch

import wandb
from modsim2.model.resnet import ResnetModel


def download_model(
    model_name: str,
    folder: Optional[str] = "turing-arc/ms2/",  # TODO: argparse in script
    version: Optional[str] = ":latest",  # TODO: argparse in script
) -> ResnetModel:
    # Download the model
    run = wandb.init()
    artifact = run.use_artifact(folder + model_name + version, type="model")
    artifact_dir = artifact.download()
    run.finish()

    # Read in the model
    path = os.path.join(artifact_dir, "model.ckpt")
    checkpoint = torch.load(path, map_location=torch.device("cpu"))
    model = ResnetModel()
    model.load_state_dict(checkpoint["state_dict"])

    return model
