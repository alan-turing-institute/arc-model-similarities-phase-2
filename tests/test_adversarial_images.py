import functools

import torch

from modsim2.attack.adversarial_images import select_best_attack


def test_best_attack_selection():
    # A test to ensure that 'best' selection will occur. The following objects
    # simulate the outputs of attack functions in foolbox:

    # a list of epsilons out of the correct order
    epsilons = [0.5, 0.3, 0.0, 1.0]
    num_epsilon = len(epsilons)

    # a list of 4 random torch.tensors to represent the 'images'
    num_image = 16
    image_shape = (num_image, 3, 32, 32)
    images = [
        torch.rand(image_shape) for i in range(num_epsilon)
    ]  # for the test, values don't matter

    # a tensor of successes, where all epsilons above 0 are successful
    success = torch.zeros((num_epsilon, num_image), dtype=torch.bool)
    success[[0, 1, 3], :] = True

    # this means the best selection function should *always* select the
    # second image in the list (as this corresponds to the lowest epsilon 0.3
    # that is still successful)
    best_images, success_rates = select_best_attack(
        images=images, success=success, epsilons=epsilons
    )

    # Check that second image is always chosen
    assert_equal = functools.partial(torch.testing.assert_close, rtol=0, atol=0)
    assert_equal(images[1], best_images)

    # Success rates should be 0, 1, 1, 1 (as fn will sort to this order and all
    # from second image onwards are successful)
    assert_equal(success_rates, torch.tensor([0.0, 1.0, 1.0, 1.0]))
