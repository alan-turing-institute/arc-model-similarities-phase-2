from collections import Counter
from math import isclose

from modsim2.data.loader import DMPair


def test_cifar_pair_n_obs():
    drop = 0.1
    dmpair = DMPair(drop_A=drop, drop_B=0)
    dl_A = dmpair.A.train_dataloader()
    dl_B = dmpair.B.train_dataloader()
    assert len(dl_A.dataset) == (1 - drop) * len(dl_B.dataset)


def test_cifar_pair_stratification():
    drop = 0.1
    dmpair = DMPair(drop_A=drop)

    # Extract labels from full and subsetted dataset
    full_labels = [i[1] for i in dmpair.A.dataset_train.dataset]
    subset_labels = [
        dmpair.A.dataset_train.dataset.dataset.targets[i]
        for i in dmpair.A.dataset_train.indices
    ]

    # Get count of labels in each dataset
    full_count = Counter(full_labels)
    subset_count = Counter(subset_labels)

    # Check that the subset count is (1-drop%)*full count, allowing for rounding error
    count_test = [
        isclose(full_count[i] * (1 - drop), subset_count[i], abs_tol=1)
        for i in range(9)
    ]
    assert all(count_test)


def test_cifar_pair_overlap():
    # Generate datasets A and B. Drop 10% from both with no overlap
    drop = 0.1
    dmpair = DMPair(drop_A=drop, drop_B=drop)
    dl_A = dmpair.A.train_dataloader()
    dl_B = dmpair.B.train_dataloader()

    # Get indices
    indices_A = set(dl_A.dataset.indices)
    indices_B = set(dl_B.dataset.indices)

    # Since no overlap in data dropped, if we drop x obs, both shld contain
    # N-x obs, with x obs not in the other. Shared obs should therefore be equal
    # to N-2x
    assert len(indices_A & indices_B) == len(dl_A.dataset.dataset) * (1 - drop * 2)
