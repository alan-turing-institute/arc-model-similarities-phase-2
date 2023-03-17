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
) -> torch.tensor:
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
    num_attack_images = len(images)
    for i in range(num_attack_images):
        for j in range(num_epsilon):
            if success[j][i] or j == (num_epsilon - 1):
                advs_images.append(clipped_advs[j][i])
                break

    # Return images
    return torch.stack((advs_images))


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
    This function loops over a 2*2 combination of models (A and B) and image/label
    pairs (A and B) to produce 4 sets of adversial images.

    The goal is to train a set of adversial images on each model, drawing the images
    from the respective distributions of A and B.

    The function returns a flat dictionary where each key:element pair is a torch.tensor
    of the results of generate_adversial_images(). The key names follow a pattern of
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

    Returns: dict[torch.tensor]: A dictionary of 4 sets of adverisal images
    """
    # Get num of images
    num_attack_images = len(images_A)

    # Make dict of adversial images
    # 4 elements, w/ keys like model_A_dist_A
    # Each element is a torch.tensor containing adversial images
    adversial_images_dict = {}
    for model in [(model_A, "model_A"), (model_B, "model_B")]:
        for pair in [(images_A, labels_A, "dist_A"), (images_B, labels_B, "dist_B")]:
            adversial_images_dict[f"{model[1]}_{pair[2]}"] = generate_adversial_images(
                model=model[0],
                images=pair[0],
                labels=pair[1],
                num_attack_images=num_attack_images,
                attack_fn_name=attack_fn_name,
                **kwargs,
            )

    # Return
    return adversial_images_dict
