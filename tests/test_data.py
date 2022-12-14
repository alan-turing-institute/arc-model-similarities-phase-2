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

    full_labels = [i[1] for i in cifar.dataset_train]
    subset_labels = [dl.dataset.dataset.dataset.targets[i] for i in dl.dataset.indices]

    full_count = Counter(full_labels)
    subset_count = Counter(subset_labels)

    count_test = [
        isclose(full_count[i] * (1 - drop), subset_count[i], rel_tol=1)
        for i in range(9)
    ]
    assert all(count_test)
