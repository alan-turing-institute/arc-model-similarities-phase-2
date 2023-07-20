import foolbox as fb
import torch
from tqdm import tqdm

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


def generate_adversarial_images(
    model: ResnetModel,
    images: torch.tensor,
    labels: torch.tensor,
    attack_fn_name: str,
    epsilons: list[float],
    device: str,
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
        **kwargs: Additional arguments based to attack setup

    Returns: a torch.tensor containing the adversarial images and a torch.tensor
             where each element represents the percentage of successful attacks at
             each value of epsilon
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

    # Make sure images are on correct device
    if str(images.device) != device:
        images = images.to(device=device)
    if str(labels.device) != device:
        labels = labels.to(device=device)

    # Put model into foolbox format
    fmodel = fb.PyTorchModel(model, bounds=(0, 1), device=device)

    # Generate attack images
    attack = getattr(fb.attacks, attack_fn_name)(**kwargs)
    _, clipped_advs, success = attack(fmodel, images, labels, epsilons=epsilons)

    # Apply image selection based on attack choice
    advs_images, advs_success = select_best_attack(
        images=clipped_advs, success=success, epsilons=epsilons
    )

    # Return images
    return advs_images, advs_success


def generate_over_combinations(
    model_A: ResnetModel,
    model_B: ResnetModel,
    images_A: torch.tensor,
    labels_A: torch.tensor,
    images_B: torch.tensor,
    labels_B: torch.tensor,
    attack_fn_name: str,
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
        **kwargs: Arguments passed to the attack function

    Returns: A dictionary of 4 sets of adverisal images
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
                **kwargs,
            )

    # Return
    return adversarial_images_dict
