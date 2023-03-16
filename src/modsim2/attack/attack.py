import foolbox as fb
import torch

from modsim2.model.resnet import ResnetModel


def generate_adversial_images(
    model: ResnetModel,
    images: torch.tensor,
    labels: list[int],
    num_attack_images: int,
    attack_fn_name: str,
    epsilons: list[float],
    **kwargs,
):
    """
    Generates adversial images from bias images for either an L2 fast gradient
    attack or a boundary attack.

    Args:
        model: Resnet model object to train the adversial images on
        images: torch.tensor of base images to build the adversial images on
        labels: correct labels for each of the images
        attack_fn_name: Name of the attack function in foolbox.attacks. Must be
                        L2FastGradientAttack or BoundaryAttack
        epsilons: Pertubation parameter for the attacks
        **kwargs: Additional arguments based to attack setup

    Returns: a torch.tensor containing the adversial images
    """
    # Check for valid attack choices
    if attack_fn_name not in ["L2FastGradientAttack", "BoundaryAttack"]:
        raise NotImplementedError("This attack has not been implemented")

    # Put model into foolbox format
    fmodel = fb.PyTorchModel(model, bounds=(0, 1))

    # Generate attack images
    attack = getattr(fb.attacks, attack_fn_name)(**kwargs)
    _, clipped_advs, success = attack(fmodel, images, labels, epsilons=epsilons)

    # Apply image selection based on attack choice
    advs_images = []
    num_epsilon = len(epsilons)
    num_attack_images = len()
    for i in range(num_attack_images):
        for j in range(num_epsilon):
            if success[j][i] or j == (num_epsilon - 1):
                advs_images.append(clipped_advs[j][i])
                break

    # Return images
    return torch.stack((advs_images))
