from collections import Counter
from math import isclose

from modsim2.data.loader import CIFAR10DataModuleDrop


def test_drop_loader_n_obs():
    drop = 0.1
    cifar = CIFAR10DataModuleDrop(drop=drop)
    cifar.prepare_data()
    cifar.setup()
    dl = cifar.train_dataloader()
    assert len(dl.dataset) == (1 - drop) * len(cifar.dataset_train)


def test_drop_loader_stratification():
    drop = 0.1
    cifar = CIFAR10DataModuleDrop(drop=drop)
    cifar.prepare_data()
    cifar.setup()
    dl = cifar.train_dataloader()

    # Extract labels from full and subsetted dataset
    full_labels = [i[1] for i in cifar.dataset_train]
    subset_labels = [dl.dataset.dataset.dataset.targets[i] for i in dl.dataset.indices]

    # Get count of labels in each dataset
    full_count = Counter(full_labels)
    subset_count = Counter(subset_labels)

    # Check that the subset count is (1-drop%)*full count, allowing for rounding error
    count_test = [
        isclose(full_count[i] * (1 - drop), subset_count[i], abs_tol=1)
        for i in range(9)
    ]
    assert all(count_test)
