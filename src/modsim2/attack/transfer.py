from typing import Callable

import torch
from pytorch_lightning import Trainer

from modsim2.model.resnet import ResnetModel


def compute_success_rate(
    labels: torch.tensor,
    base_correct: torch.tensor,
    advs_preds: torch.tensor,
) -> float:
    """
    A function that computes the success rate of transfer attacks, is given by the
    percentage of images which in their base form are correctly identified by the
    network but which in their adversial form are not.

    Args:
        labels: The correct labels
        base_correct: List denoting which base images the network correctly
                      predicts
        advs_preds: List of predictions for the adversial images
    """
    advs_correct = (labels == advs_preds)[base_correct]
    success_rate = torch.sum(advs_correct) / torch.sum(base_correct)
    return success_rate


def compute_mean_loss_rate(
    labels: torch.tensor,
    base_loss: torch.tensor,
    advs_softmax: torch.tensor,
    loss_function: Callable,
) -> float:
    """
    A function that computes the mean loss rate of transfer attacks. This is given by
    the mean increase in loss of all images from base to adversial.

    Args:
        labels: The correct labels
        base_correct: List denoting which base images the network correctly
                      predicts
        advs_preds: List of predictions for the adversial images
        loss_function: Loss function to use in computing mean_loss_rate
    """
    advs_loss = loss_function(advs_softmax, labels)
    mean_loss_rate = advs_loss - base_loss
    return mean_loss_rate


def compute_transfer_attack(
    model: ResnetModel,
    images: torch.tensor,
    labels: torch.tensor,
    advs_images: list[torch.tensor],
    attack_names: str,
    batch_size: int,
    loss_function: Callable = torch.nn.functional.nll_loss,
) -> dict[dict[float]]:
    """
    This function takes a model, base images, adversial images, and true labels
    as inputs, and attacks the model. Where the images were generated on a
    different model, it will be a transfer attack.

    As output, this function will compute the attack success rate and mean loss
    increase as metrics and provide them in a dictionary.

    Attack success rate is given by the percentage of images which in their base
    form are correctly identified by the network but which in their adversial form
    are not.

    Mean loss increase is simply the mean increase in loss of all images from base
    to adversial.

    Args:
        model: The model to transfer the attacks to
        images: A torch.tensor of base images
        labels: A list of correct labels corresponding to each image
        advs_images: A torch.tensor of adversial images, which corresponds to the base
                     images
        attack_names: Output strings for the attack names
        batch_size: Batch size for the dataloader used in predicting outputs
        loss_function: Loss function to use in computing mean_loss_rate

    Returns: a dictionary of attack success metrics
    """
    # Generate base image and attack image softmax and predictions
    # 128
    images_dl = torch.utils.data.DataLoader(
        images, batch_size=batch_size, shuffle=False, sampler=None
    )
    trainer = Trainer()
    base_softmax = torch.cat(trainer.predict(model, images_dl))
    base_preds = torch.max(base_softmax, dim=1)[1]

    # These are recycled for each attack, worth computing now
    base_correct = labels == base_preds
    base_loss = loss_function(base_softmax, labels)

    # Generate adversarial softmax and predictions
    advs_dl = [
        torch.utils.data.DataLoader(
            images, batch_size=batch_size, shuffle=False, sampler=None
        )
        for images in advs_images
    ]
    advs_softmax = [torch.cat(trainer.predict(model, dl)) for dl in advs_dl]
    advs_preds = [torch.max(softmax, dim=1)[1] for softmax in advs_softmax]

    # Compute transfer attack success metrics
    transfer_metrics = {}

    # Loop over attacks, compute metrics for each attack
    for index, attack_name in enumerate(attack_names):
        # Dict to store metrics specific to that attack
        transfer_metrics[attack_name] = {}

        # Success rate
        transfer_metrics[attack_name]["success_rate"] = compute_success_rate(
            labels=labels,
            base_correct=base_correct,
            advs_preds=advs_preds[index],
        )

        # Mean loss rate
        transfer_metrics[attack_name]["mean_loss_increase"] = compute_mean_loss_rate(
            labels=labels,
            base_loss=base_loss,
            advs_softmax=advs_softmax[index],
            loss_function=loss_function,
        )

    # Return
    return transfer_metrics
