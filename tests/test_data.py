from collections import Counter
from copy import deepcopy
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


def test_drop_loader_double_overlap():
    # Generate datasets A and B. Drop 10% from both with no overlap
    drop = 0.1
    cifar_A = CIFAR10DataModuleDrop(drop=drop)
    cifar_B = deepcopy(cifar_A)
    cifar_B.keep = "B"

    # Set up the datasets
    cifar_A.prepare_data()
    cifar_A.setup()
    cifar_B.prepare_data()
    cifar_B.setup()
    dl_A = cifar_A.train_dataloader()
    dl_B = cifar_B.train_dataloader()

    # Extract indices from both datasets to work as lists
    indices_A = set(dl_A.dataset.indices)
    indices_B = set(dl_B.dataset.indices)

    # Since no overlap in data dropped, if we drop x obs, both shld contain
    # N-x obs, with x obs not in the other. Shared obs should therefore be equal
    # to N-2x
    assert len(indices_A & indices_B) == len(dl_A.dataset.dataset) * (1 - drop * 2)
