from collections import Counter
from math import ceil, isclose

import testing_constants

from modsim2.data.loader import CIFAR10DMSubset, DMPair


def _test_dm_n_obs(
    datamodule_subset: CIFAR10DMSubset,
    drop: float,
    val_split: float,
) -> None:
    orig_train_size = testing_constants.DUMMY_CIFAR10_SIZE * (1 - val_split)
    orig_val_size = testing_constants.DUMMY_CIFAR10_SIZE * val_split
    assert len(datamodule_subset.dataset_train) == round((1 - drop) * orig_train_size)
    assert len(datamodule_subset.dataset_val) == round((1 - drop) * orig_val_size)


def test_cifar_empty_val():
    drop_A = 0.2
    val_split = 0
    dmpair = DMPair(drop_percent_A=drop_A, val_split=val_split)
    # test A
    _test_dm_n_obs(datamodule_subset=dmpair.A, drop=drop_A, val_split=val_split)
    # test B
    _test_dm_n_obs(datamodule_subset=dmpair.B, drop=0.0, val_split=val_split)


def test_cifar_A_n_obs():
    drop_A = 0.2
    val_split = 0.4
    dmpair = DMPair(drop_percent_A=drop_A, val_split=val_split)
    # test A
    _test_dm_n_obs(datamodule_subset=dmpair.A, drop=drop_A, val_split=val_split)
    # test B
    _test_dm_n_obs(datamodule_subset=dmpair.B, drop=0.0, val_split=val_split)


def test_cifar_B_n_obs():
    drop_B = 0.3
    val_split = 0.4
    dmpair = DMPair(drop_percent_B=drop_B, val_split=val_split)
    # test A
    _test_dm_n_obs(datamodule_subset=dmpair.A, drop=0.0, val_split=val_split)
    # test B
    _test_dm_n_obs(datamodule_subset=dmpair.B, drop=drop_B, val_split=val_split)


def test_cifar_both_n_obs():
    drop_A = 0.2
    drop_B = 0.3
    val_split = 0.4
    dmpair = DMPair(drop_percent_A=drop_A, drop_percent_B=drop_B, val_split=val_split)
    # test A
    _test_dm_n_obs(datamodule_subset=dmpair.A, drop=drop_A, val_split=val_split)
    # test B
    _test_dm_n_obs(datamodule_subset=dmpair.B, drop=drop_B, val_split=val_split)


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
    drop_A = 0.2
    drop_B = 0.3
    val_split = 0.4
    dmpair = DMPair(drop_percent_A=drop_A, drop_percent_B=drop_B, val_split=val_split)
    full_train_labels = [image[1] for image in dmpair.cifar.dataset_train]
    full_val_labels = [image[1] for image in dmpair.cifar.dataset_val]
    # test A
    train_labels_A, val_labels_A = dmpair.get_A_labels()
    train_labels_B, val_labels_B = dmpair.get_B_labels()
    _test_dm_stratification(full_train_labels, train_labels_A, drop=drop_A)
    # test B
    _test_dm_stratification(full_train_labels, train_labels_B, drop=drop_B)

    # val
    # test A
    _test_dm_stratification(full_val_labels, val_labels_A, drop=drop_A)
    # test B
    _test_dm_stratification(full_val_labels, val_labels_B, drop=drop_B)


def _test_dm_overlap_split(
    indices_A: list[int],
    indices_B: list[int],
    drop_percentage_A: float,
    drop_percentage_B: float,
    orig_total_size: int,
) -> None:
    # Get indices, put into set
    indices_A = set(indices_A)
    indices_B = set(indices_B)

    # Below is based on notion that there should be no overlap in data
    # dropped from both datasets
    total_drop_percentage = drop_percentage_A + drop_percentage_B
    shared_size = orig_total_size * (1 - total_drop_percentage)
    assert len(indices_A & indices_B) == shared_size


def _test_dm_overlap(drop_A: float, drop_B: float, val_split: float):
    dmpair = DMPair(drop_percent_A=drop_A, drop_percent_B=drop_B, val_split=val_split)

    orig_train_size = round(testing_constants.DUMMY_CIFAR10_SIZE * (1 - val_split))
    orig_val_size = testing_constants.DUMMY_CIFAR10_SIZE - orig_train_size

    _test_dm_overlap_split(
        dmpair.A.dataset_train.indices,
        dmpair.B.dataset_train.indices,
        dmpair.drop_percent_A,
        dmpair.drop_percent_B,
        orig_total_size=orig_train_size,
    )

    _test_dm_overlap_split(
        dmpair.A.dataset_val.indices,
        dmpair.B.dataset_val.indices,
        dmpair.drop_percent_A,
        dmpair.drop_percent_B,
        orig_total_size=orig_val_size,
    )


def test_cifar_pair_overlap_same_size():
    drop = 0.2
    val_split = 0.4
    _test_dm_overlap(drop_A=drop, drop_B=drop, val_split=val_split)


def test_cifar_pair_overlap_diff_size():
    drop_A = 0.2
    drop_B = 0.3
    val_split = 0.4
    _test_dm_overlap(drop_A=drop_A, drop_B=drop_B, val_split=val_split)


def _test_cifar_dataloader_batch_count(data_loader, batch_size):
    dataset_size = len(data_loader.dataset)
    count = 0
    for _ in data_loader:
        count += 1
    assert count == ceil(dataset_size / batch_size)


def test_cifar_A_batch_count():
    batch_size = 32
    val_split = 0.4
    dmpair = DMPair(drop_percent_A=0.175, batch_size=batch_size, val_split=val_split)
    _test_cifar_dataloader_batch_count(dmpair.A.train_dataloader(), batch_size)
    _test_cifar_dataloader_batch_count(dmpair.A.val_dataloader(), batch_size)


def test_cifar_B_batch_count():
    batch_size = 32
    val_split = 0.4
    dmpair = DMPair(drop_percent_B=0.175, batch_size=batch_size, val_split=val_split)
    _test_cifar_dataloader_batch_count(dmpair.B.train_dataloader(), batch_size)
    _test_cifar_dataloader_batch_count(dmpair.B.val_dataloader(), batch_size)
