import torch

from modsim2.model.resnet import ResnetModel


def compute_transfer_attack(
    model: ResnetModel,
    images: torch.tensor,
    labels: list[int],
    advs_images: torch.tensor,
):
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

    Returns: a dictionary of attack success metrics
    """
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
    base_loss = torch.nn.functional.nll_loss(
        base_softmax, labels
    )  # TODO: add convinience method to model
    advs_loss = torch.nn.functional.nll_loss(advs_softmax, labels)
    transfer_metrics["mean_loss_increase"] = advs_loss - base_loss

    # Return
    return transfer_metrics