import os
from typing import Optional

import foolbox as fb
import torch
import torch.nn.functional as F
import yaml
from pytorch_lightning import LightningDataModule
from utils import opts2dmpairArgs

import wandb
from modsim2.data.loader import DMPair
from modsim2.model.resnet import ResnetModel

# TODO: delete this, add argparse to this script, make generation script for calls
project_path = "/Users/pswatton/Documents/Code/arc-model-similarities-phase-2/"
dataset_config_path = project_path + "configs/datasets.yaml"
experiment_group = "drop-only"
seed_index = 0
dataset_index = 1


# TODO: put this in src
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


# TODO: consider moving to src
def get_transfer_images(
    dl: LightningDataModule, num_images: int = 16
) -> tuple[torch.tensor, torch.tensor]:
    # Get images from test dataset
    iter_dl = iter(dl())
    images, labels = next(iter_dl)

    # Cut down the number from 32
    images = images[:num_images]
    labels = labels[:num_images]

    return images, labels


# TODO: move to src? unsure on this one. base on Marcos' code?
def generate_attacks(
    model: ResnetModel,
    images,
    labels,
    num_attack_images,
):
    # Put model into foolbox
    # TODO: check if I need to input preprocessing
    fmodel = fb.PyTorchModel(model, bounds=(0, 1))

    # Generate attack images
    # TODO: switch to taking a dict of attacks and params
    attack = fb.attacks.L2FastGradientAttack()
    epsilons = [
        0.0,
        0.1,
        0.3,
        0.5,
        1.0,
    ]
    _, clipped_advs, success = attack(fmodel, images, labels, epsilons=epsilons)

    # Filter for minimally successful case
    # Break condition means can't use list comprehension
    advs_images = []
    num_epsilon = len(epsilons)
    for i in range(num_attack_images):
        for j in range(num_epsilon):
            if success[j][i] or j == (num_epsilon - 1):
                advs_images.append(clipped_advs[j][i])
                break

    # Return images
    return torch.stack((advs_images))


# TODO: move to src/
def compute_transfer_attack(
    model: ResnetModel,
    images,
    labels,
    advs_images,
    num_attack_images: int,
):
    # Generate base image and attack image predictions
    base_softmax = model.forward(images)  # TODO
    advs_softmax = model.forward(advs_images)  # TODO: 2x
    base_preds = torch.max(base_softmax, dim=1)[1]
    advs_preds = torch.max(advs_softmax, dim=1)[1]

    # Compute transfer attack success metrics
    transfer_metrics = {}

    # Success rate
    base_correct = labels == base_preds
    advs_correct = (labels == advs_preds)[base_correct]
    transfer_metrics["success_rate"] = torch.sum(advs_correct) / torch.sum(base_correct)

    # Mean loss rate
    base_loss = F.nll_loss(
        base_softmax, labels
    )  # TODO: add convinience method to model
    advs_loss = F.nll_loss(advs_softmax, labels)
    transfer_metrics["mean_loss_increase"] = advs_loss - base_loss

    # Return
    return transfer_metrics


def main(
    dataset_config: str,
    experiment_group: str,
    dataset_index: int,
    seed_index: int,
):

    # Generate strings
    experiment_pair_name = f"{experiment_group}_{dataset_index}_{seed_index}"
    model_name_A = experiment_pair_name + "_A_model"
    model_name_B = experiment_pair_name + "_B_model"

    # Download and prepare models
    model_A = download_model(model_name_A)
    model_B = download_model(model_name_B)
    model_A.eval()
    model_B.eval()

    # Prepare dmpair
    dmpair_kwargs = opts2dmpairArgs(
        opt=dataset_config["experiment_groups"][experiment_group][dataset_index],
        seed=dataset_config["seeds"][seed_index],
    )
    dmpair = DMPair(**dmpair_kwargs)

    # Prepare test dataloaders
    dmpair.A.setup()
    dmpair.B.setup()
    test_dl_A = dmpair.A.test_dataloader
    # test_dl_B = dmpair.B.test_dataloader

    # Get transfer images
    num_attack_images = 16  # TODO: make optional argparse kwarg
    images_A, labels_A = get_transfer_images(test_dl_A, num_attack_images)
    # images_B, labels_B = get_transfer_images(test_dl_B, num_attack_images)

    # Generate attack images
    # TODO: make this a loop over combos of A and B models/images
    model_A_dl_A_attacks = generate_attacks(
        model=model_A,
        images=images_A,
        labels=labels_A,
        num_attack_images=num_attack_images,
    )

    # Transfer the attack, compute metrics based on results
    transfer_metrics_AA_to_B = compute_transfer_attack(
        model=model_B,
        images=images_A,
        labels=labels_A,
        advs_images=model_A_dl_A_attacks,
        num_attack_images=num_attack_images,
    )

    # Output
    print(transfer_metrics_AA_to_B)


if __name__ == "__main__":
    # TODO: set this up via argparse

    # with open(args.dataset_config, "r") as stream:
    with open(dataset_config_path, "r") as stream:
        dataset_config = yaml.safe_load(stream)

    main(
        dataset_config=dataset_config,
        experiment_group=experiment_group,
        dataset_index=dataset_index,
        seed_index=seed_index,
    )
