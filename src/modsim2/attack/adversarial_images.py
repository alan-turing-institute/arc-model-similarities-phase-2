import copy
from typing import Callable

import foolbox as fb
import torch
from tqdm import tqdm

from modsim2.attack.transfer import compute_transfer_attack
from modsim2.model.resnet import ResnetModel
from modsim2.utils.accelerator import choose_auto_accelerator


def select_best_attack(
    images: list[torch.tensor],
    success: torch.tensor,
    epsilons: list[float],
) -> tuple[torch.tensor, torch.tensor]:
    """
    A function to perform best attack selection. The criteria for each images is
    the first successful attack with the smallest possible value of epsilon. If no
    image is sucessful, it will select the image with the highest epsilon value.

    Args:
        images: A list of torch.tensors of length num_epsilon. Each tensor contains
                the adversarial images generated for the corresponding value of
                epsilon in the list epsilons.
        success: A torch.tensor of shape num_epsilon x num_images, containing True and
                 False values denoting whether the corresponding image in images is
                 successfully adversarial
        epsilons: A list of values of epsilon used to generate the list of images

    Returns: a torch.tensor containing the adversarial images and a torch.tensor
             where each element represents the percentage of successful attacks at
             each value of epsilon
    """
    # Sort into order from lowest to highest epsilon to allow arbitary input order
    num_epsilon = len(epsilons)
    sort_indices = sorted(range(num_epsilon), key=lambda k: epsilons[k])
    epsilons = [epsilons[index] for index in sort_indices]
    success = success[sort_indices, :]
    images = [images[index] for index in sort_indices]

    # Best attack selection loop
    advs_images = []
    advs_success = torch.zeros(num_epsilon, device=success.device)
    num_attack_images = len(images[0])
    # For every image and value of epsilon
    for i in range(num_attack_images):
        for j in range(num_epsilon):
            # If the image is sucessful or is the last image, append it to the
            # output adversarial images
            if success[j][i] or j == (num_epsilon - 1):
                advs_images.append(images[j][i])
                # += success instead of 1 so last image doesn't automatically add
                # to success
                advs_success[j:] += success[j][i]
                break
    return torch.stack((advs_images)), advs_success / num_attack_images


def boundary_attack_fn(
    fmodel,
    images,
    labels,
    attack_fn_name,
    epsilons,
    device,
    **kwargs,
) -> tuple[list[torch.tensor], torch.tensor]:
    """
    A function that manually initialises an attack, filters for successful
    initialisations using foolbox's own tests, and uses these to generate
    adversarial images. Written for Boundary Attacks, may be valid to use
    for other attack types too.

    Images not succesfully initialised are return in the same output as adversarial
    examples, and marked as failures in the same way other failures would be.

    Args:
        model: Resnet model object to train the adversarial images on
        images: torch.tensor of base images to build the adversarial images on
        labels: correct labels for each of the images
        attack_fn_name: Name of the attack function in foolbox.attacks.
        epsilons: Pertubation parameter for the attacks
        device: String passed to fb.PyTorchModel for computation
        **kwargs: Additional arguments based to attack setup

    Returns:
        A list where each element corresponds to a value of epsilon. Each element
        contains a torch.tensor of adversarial examples corresponding to
    """
    # Initialise and run an attack
    init_attack = fb.attacks.LinearSearchBlendedUniformNoiseAttack(
        distance=fb.distances.LpDistance(p=2), steps=50
    )
    starting_points = init_attack.run(fmodel, images, labels)

    # Assess whether the attack is adversarial, get tensor of successes
    is_adversarial = fb.attacks.base.get_is_adversarial(
        fb.attacks.base.get_criterion(labels), fmodel
    )
    starting_success = is_adversarial(starting_points)

    # Generate attack images on already successful adversarial examples
    attack = getattr(fb.attacks, attack_fn_name)(**kwargs)
    _, clipped_advs, success = attack(
        fmodel,
        images[starting_success],
        labels[starting_success],
        starting_points=starting_points[starting_success],
        epsilons=epsilons,
    )

    # Get indices for examples kept
    starting_indices = [
        i for i, _ in enumerate(starting_success) if starting_success[i]
    ]

    # Prepare new set of images and successes
    new_advs = [copy.deepcopy(images) for _ in epsilons]
    new_success = torch.zeros(
        (len(epsilons), images.shape[0]), dtype=torch.bool, device=device
    )

    # Replace original images and successes with ones from the attack
    for i in range(len(epsilons)):
        new_advs[i][starting_indices] = clipped_advs[i]
        new_success[i][starting_indices] = success[i]

    # Return
    return new_advs, new_success


def generate_adversarial_images(
    model: ResnetModel,
    images: torch.tensor,
    labels: torch.tensor,
    attack_fn_name: str,
    epsilons: list[float],
    device: str,
    batch_size: int,
    trainer_kwargs: dict = {},
    loss_function: Callable = torch.nn.functional.nll_loss,
    **kwargs,
) -> tuple[torch.tensor, torch.tensor]:
    """
    Generates adversarial images from bias images for either an L2 fast gradient
    attack or a boundary attack.

    Args:
        model: Resnet model object to train the adversarial images on
        images: torch.tensor of base images to build the adversarial images on
        labels: correct labels for each of the images
        attack_fn_name: Name of the attack function in foolbox.attacks. Must be
                        L2FastGradientAttack or BoundaryAttack
        epsilons: Pertubation parameter for the attacks
        device: String passed to fb.PyTorchModel for computation
        batch_size: int,
        trainer_kwargs: dict = {},
        loss_function: Callable = torch.nn.functional.nll_loss,
        **kwargs: Additional arguments based to attack setup

    Returns: a torch.tensor containing the adversarial images and a ductionary
             where each element is a model vulnerability metric. These are
             'success_rate' and 'mean_loss_increase' respectively.

    """
    # Check for valid attack choices
    if attack_fn_name not in ["L2FastGradientAttack", "BoundaryAttack"]:
        raise NotImplementedError("This attack has not been implemented")

    # If needed, auto select pytorch device/accelerator
    if device == "auto":
        device = choose_auto_accelerator()
    # Issue #44: Depenency chain leads to float64 error on MPS
    if device == "mps" and attack_fn_name == "BoundaryAttack":
        device = "cpu"

    # Make sure images + model are on correct device
    if str(images.device) != device:
        images = images.to(device=device)
    if str(labels.device) != device:
        labels = labels.to(device=device)
    if str(model.device) != device:
        model = model.to(device=device)

    # Put model into foolbox format
    fmodel = fb.PyTorchModel(model, bounds=(0, 1), device=device)

    # If Boundary Attack, need to initialise and manage early failures to avoid error
    if attack_fn_name == "BoundaryAttack":
        clipped_advs, success = boundary_attack_fn(
            fmodel=fmodel,
            images=images,
            labels=labels,
            attack_fn_name=attack_fn_name,
            epsilons=epsilons,
            device=device,
            **kwargs,
        )

    # If not Boundary Attack, generate images as normal
    if attack_fn_name != "BoundaryAttack":
        attack = getattr(fb.attacks, attack_fn_name)(**kwargs)
        _, clipped_advs, success = attack(fmodel, images, labels, epsilons=epsilons)

    # Apply image selection based on attack choice
    advs_images, _ = select_best_attack(
        images=clipped_advs, success=success, epsilons=epsilons
    )

    # Get the model vulnerability metrics
    vuln = compute_transfer_attack(
        model=model,
        images=images,
        labels=labels,
        advs_images=[advs_images],
        attack_names=[attack_fn_name],
        batch_size=batch_size,
        loss_function=loss_function,
        devices="auto",
        accelerator=device,
        **trainer_kwargs,
    )
    vuln = {**vuln[attack_fn_name]}

    # Return images
    return advs_images, vuln


def generate_over_combinations(
    model_A: ResnetModel,
    model_B: ResnetModel,
    images_A: torch.tensor,
    labels_A: torch.tensor,
    images_B: torch.tensor,
    labels_B: torch.tensor,
    attack_fn_name: str,
    batch_size: int,
    trainer_kwargs: dict = {},
    loss_function: Callable = torch.nn.functional.nll_loss,
    **kwargs,
) -> dict[torch.tensor]:
    """
    This function loops over models and images to produce the four sets of adversarial
    attacks. These combinations are:

    model A, images A
    model A, images B
    model B, images A
    model B, images B

    The goal is to train a set of adversarial images on each model, drawing the images
    from the respective distributions of A and B.

    The function returns a flat dictionary where each key:element pair is a torch.tensor
    of the results of generate_adversarial_images(). The key names follow a pattern of
    model_A_dist_A for model A and images from the distribution of A.

    Args:
        model_A: ResNet model trained on dataset A
        model_B: ResNet model trained on dataset B
        images_A: torch.tensor of images from the distribution of dataset A
        labels_A: torch.tensor of labels corresponding to images_A
        images_B: torch.tensor of images from the distribution of dataset B
        labels_B: torch.tensor of labels corresponding to images_B
        attack_fn_name: String corresponding to the attack function to use
        batch_size: Batch size for the dataloader used in predicting outputs
        loss_function: Loss function to use in computing mean_loss_rate
        trainer_kwargs: Keyword arguments passed to pytorch_lightning.Trainer()
        **kwargs: Arguments passed to the attack function

    Returns: A dictionary of 4 sets of adverisal images with associated model
             vulnerability metrics
    """
    # Make dict of adversarial images
    # 4 elements, w/ keys like model_A_dist_A
    # Each element is a torch.tensor containing adversarial images and a
    # torch.tensor with success rates at each value of epsilon
    adversarial_images_dict = {}
    for model, model_name in tqdm([(model_A, "model_A"), (model_B, "model_B")]):
        for images, labels, distribution_name in tqdm(
            [
                (images_A, labels_A, "dist_A"),
                (images_B, labels_B, "dist_B"),
            ]
        ):
            adversarial_images_dict[
                f"{model_name}_{distribution_name}"
            ] = generate_adversarial_images(
                model=model,
                images=images,
                labels=labels,
                attack_fn_name=attack_fn_name,
                batch_size=batch_size,
                loss_function=loss_function,
                trainer_kwargs=trainer_kwargs,
                **kwargs,
            )

    # Return
    return adversarial_images_dict
