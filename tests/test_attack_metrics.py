import torch

from modsim2.attack.metrics import compute_mean_loss_rate, compute_success_rate


def test_success_rate():
    # Simple test case - indices 4 and 5 should always be ignored by the tests
    # as sucess rate depens only on those adversarial images where the base images
    # is correctly classified by the network
    labels = torch.tensor([0, 0, 1, 1, 0, 1])
    base_correct = torch.tensor([True, True, True, True, False, False])

    # Test where adversarial transfer success should be 0
    advs_preds_same = torch.tensor([0, 0, 1, 1, 0, 0])
    assert compute_success_rate(labels, base_correct, advs_preds_same) == 0

    # Test where adversarial transfer success should be 0.5
    advs_preds_half = torch.tensor([0, 1, 0, 1, 0, 0])
    assert compute_success_rate(labels, base_correct, advs_preds_half) == 0.5

    # Test where adversarial transfer success should be 1
    advs_preds_all = torch.tensor([1, 1, 0, 0, 0, 0])
    assert compute_success_rate(labels, base_correct, advs_preds_all) == 1


def test_mean_loss_rate():
    # Simple test case
    labels = torch.tensor([0, 1])
    base_softmax = torch.tensor([[0.8, 0.2], [0.2, 0.8]])
    f = torch.nn.functional.nll_loss
    base_loss = f(base_softmax, labels)

    # Case where loss change should be 0 as softmax is identical
    assert (
        compute_mean_loss_rate(
            labels=labels,
            base_loss=base_loss,
            advs_softmax=base_softmax,
            loss_function=f,
        )
        == 0
    )

    advs_softmax_different = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    assert (
        compute_mean_loss_rate(
            labels=labels,
            base_loss=base_loss,
            advs_softmax=advs_softmax_different,
            loss_function=f,
        )
        == 0.3
    )
