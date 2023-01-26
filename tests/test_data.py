from collections import Counter
from math import ceil, isclose

from modsim2.data.loader import DMPair

# Constants for testing
VAL_SPLIT = 0.2
CIFAR_10_TRAIN_SIZE = 40000


def _test_dm_n_obs(
    length_subset: int,
    drop: float,
) -> None:
    assert length_subset == (1 - drop) * CIFAR_10_TRAIN_SIZE

def _test_cifar_dataloader_batch_count(data_loader, batch_size):
    dataset_size = len(data_loader.dataset)
    count = 0
    for batch in data_loader:
        count += 1
    assert count == ceil(dataset_size / batch_size)

def test_cifar_A_n_obs():
    # test n observations
    drop = 0.175
    dmpair = DMPair(drop_percent_A=drop, val_split=VAL_SPLIT)
    _test_dm_n_obs(len(dmpair.A.dataset_train), drop)
    # test batching
    batch_size = 32
    dla = dmpair.A.train_dataloader()
    _test_cifar_dataloader_batch_count(dla, batch_size)


def test_cifar_B_n_obs():
    # test n observations
    drop = 0.175
    dmpair = DMPair(drop_percent_B=drop, val_split=VAL_SPLIT)
    _test_dm_n_obs(len(dmpair.B.dataset_train), drop)
    # test batching
    batch_size = 32
    dlb = dmpair.B.train_dataloader()
    _test_cifar_dataloader_batch_count(dlb, batch_size)


def _test_dm_stratification(
    full_labels: list[int], subset_labels: list[int], drop: float
) -> None:
    # Get count of labels in each dataset
    full_count = Counter(full_labels)
    subset_count = Counter(subset_labels)

    # Check that the subset count is (1-drop%)*full count, allowing for rounding error
    count_test = [
        isclose(full_count[i] * (1 - drop), subset_count[i], abs_tol=1)
        for i in range(9)
    ]
    assert all(count_test)


def test_cifar_stratification():
    a_drop = 0.1
    b_drop = 0.1
    dmpair = DMPair(drop_percent_A=a_drop, drop_percent_B=b_drop)
    full_labels = [image[1] for image in dmpair.cifar.dataset_train]
    _test_dm_stratification(full_labels, dmpair.labels_A, drop=a_drop)
    _test_dm_stratification(full_labels, dmpair.labels_B, drop=b_drop)
    


def _test_dm_overlap(
    indices_A: list[int],
    indices_B: list[int],
    drop_percentage_A: float,
    drop_percentage_B: float,
) -> None:
    # Get indices, put into set
    indices_A = set(indices_A)
    indices_B = set(indices_B)

    # Below is based on notion that there should be no overlap in data
    # dropped from both datasets
    total_drop_percentage = drop_percentage_A + drop_percentage_B
    shared_size = CIFAR_10_TRAIN_SIZE * (1 - total_drop_percentage)
    assert len(indices_A & indices_B) == shared_size


def test_cifar_pair_overlap_same_size():
    drop = 0.1
    dmpair = DMPair(drop_percent_A=drop, drop_percent_B=drop, val_split=VAL_SPLIT)
    _test_dm_overlap(
        dmpair.indices_A, dmpair.indices_B, dmpair.drop_percent_A, dmpair.drop_percent_B
    )


def test_cifar_pair_overlap_diff_size():
    dmpair = DMPair(drop_percent_A=0.1, drop_percent_B=0.2, val_split=VAL_SPLIT)
    _test_dm_overlap(
        dmpair.indices_A, dmpair.indices_B, dmpair.drop_percent_A, dmpair.drop_percent_B
    )


