from collections import Counter
from math import ceil, isclose
from unittest.mock import MagicMock

import numpy as np
import testing_constants
import torch

from modsim2.data.loader import CIFAR10DMSubset, DMPair


def _test_n_obs(
    train_data: torch.Tensor, val_data: torch.Tensor, drop: float, val_split: float
):
    orig_train_size = testing_constants.DUMMY_CIFAR10_SIZE * (1 - val_split)
    orig_val_size = testing_constants.DUMMY_CIFAR10_SIZE * val_split
    assert len(train_data) == round((1 - drop) * orig_train_size)
    assert len(val_data) == round((1 - drop) * orig_val_size)


def _test_dm_n_obs(
    datamodule_subset: CIFAR10DMSubset,
    drop: float,
    val_split: float,
) -> None:
    _test_n_obs(
        train_data=datamodule_subset.dataset_train,
        val_data=datamodule_subset.dataset_val,
        drop=drop,
        val_split=val_split,
    )


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


def _setup_transforms_test(
    batch_size: int, drop_A: float = 0.0, drop_B: float = 0.0, val_split: float = 0.4
) -> tuple[DMPair, torch.tensor, torch.tensor, torch.tensor]:
    """
    setup transforms test with dmpair and expected values provided by mocks of
    transforms. Returns dmpair and mocks for train, val, test transforms
    """
    batch_size = 16

    train_output = torch.rand((3, 2, 2))
    val_output = torch.rand((3, 2, 2))
    test_output = torch.rand((3, 2, 2))

    train_transforms_mock = MagicMock(return_value=train_output)
    val_transforms_mock = MagicMock(return_value=val_output)
    test_transforms_mock = MagicMock(return_value=test_output)

    dmpair = DMPair(
        drop_percent_A=drop_A,
        drop_percent_B=drop_B,
        val_split=val_split,
        train_transforms=train_transforms_mock,
        val_transforms=val_transforms_mock,
        test_transforms=test_transforms_mock,
        batch_size=batch_size,
        drop_last=True,
    )
    dmpair.A.setup()
    dmpair.B.setup()
    return dmpair, train_transforms_mock, val_transforms_mock, test_transforms_mock


def _test_dl_transforms(
    transforms_mock: MagicMock,
    batch_size: int,
    dl: torch.utils.data.DataLoader,
    raw_orig_data: torch.Tensor,
):
    # dataloader has last batch dropped so don't need to worry
    expected_batch_train = transforms_mock.return_value.repeat(batch_size, 1, 1, 1)
    for data, labels in dl:
        # batch, channels, height, width
        assert torch.equal(data, expected_batch_train)

    # check that was called with raw image (no other transforms had happened first)
    called_im = torch.from_numpy(np.asarray(transforms_mock.call_args[0][0]))
    # check this im (last call) matches an image from raw orig - don't know order
    assert any([torch.all(called_im == img) for img in raw_orig_data])


def test_transforms():
    # check our transform runs
    batch_size = 16
    (
        dmpair,
        train_transforms_mock,
        val_transforms_mock,
        test_transforms_mock,
    ) = _setup_transforms_test(batch_size=batch_size)

    # load orig raw data - for checking transforms called with orig data
    raw_train = torch.load(testing_constants.DUMMY_CIFAR_TRAIN)
    reshaped_raw_train = raw_train[0].reshape(-1, 3, 32, 32).permute(0, 2, 3, 1)
    raw_test = torch.load(testing_constants.DUMMY_CIFAR_TEST)
    reshaped_raw_test = raw_test[0].reshape(-1, 3, 32, 32).permute(0, 2, 3, 1)

    _test_dl_transforms(
        transforms_mock=train_transforms_mock,
        batch_size=batch_size,
        dl=dmpair.A.train_dataloader(),
        raw_orig_data=reshaped_raw_train,
    )
    _test_dl_transforms(
        transforms_mock=train_transforms_mock,
        batch_size=batch_size,
        dl=dmpair.B.train_dataloader(),
        raw_orig_data=reshaped_raw_train,
    )

    _test_dl_transforms(
        transforms_mock=val_transforms_mock,
        batch_size=batch_size,
        dl=dmpair.A.val_dataloader(),
        raw_orig_data=reshaped_raw_train,
    )
    _test_dl_transforms(
        transforms_mock=val_transforms_mock,
        batch_size=batch_size,
        dl=dmpair.B.val_dataloader(),
        raw_orig_data=reshaped_raw_train,
    )

    _test_dl_transforms(
        transforms_mock=test_transforms_mock,
        batch_size=batch_size,
        dl=dmpair.A.test_dataloader(),
        raw_orig_data=reshaped_raw_test,
    )
    _test_dl_transforms(
        transforms_mock=test_transforms_mock,
        batch_size=batch_size,
        dl=dmpair.B.test_dataloader(),
        raw_orig_data=reshaped_raw_test,
    )


def _test_get_x_data(
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    raw_data: torch.Tensor,
    batch_size: int,
    drop: float,
    val_split: float,
    train_transforms_mock: MagicMock,
    val_transforms_mock: MagicMock,
):
    # test n_obs
    _test_n_obs(
        train_data=train_data, val_data=val_data, drop=drop, val_split=val_split
    )

    # test transforms
    train_data = train_data.unsqueeze(1)
    val_data = val_data.unsqueeze(1)
    # zip with fake labels to look like dataloader with batches
    train_data = zip(train_data, range(train_data.shape[0]))
    val_data = zip(val_data, range(val_data.shape[0]))

    _test_dl_transforms(
        transforms_mock=train_transforms_mock,
        batch_size=batch_size,
        dl=train_data,
        raw_orig_data=raw_data,
    )
    _test_dl_transforms(
        transforms_mock=val_transforms_mock,
        batch_size=batch_size,
        dl=val_data,
        raw_orig_data=raw_data,
    )


# test dataloader access via get_A_data and get_B_data
def test_get_AB_data():
    # same patter as test_transforms - want to check these have happened
    # also check n_obs is consistent with expectations

    # need to use batch size 1 for this test to match our non-batched iterator
    batch_size = 1
    val_split = 0.4
    drop_A = 0.2
    drop_B = 0.2
    (dmpair, train_transforms_mock, val_transforms_mock, _,) = _setup_transforms_test(
        batch_size=batch_size, drop_A=drop_A, drop_B=drop_B, val_split=val_split
    )

    # load orig raw data - for checking transforms called with orig data
    raw_train = torch.load(testing_constants.DUMMY_CIFAR_TRAIN)
    reshaped_raw_train = raw_train[0].reshape(-1, 3, 32, 32).permute(0, 2, 3, 1)

    # pad for batches
    train_data_A, val_data_A = dmpair.get_A_data()
    _test_get_x_data(
        train_data=train_data_A,
        val_data=val_data_A,
        raw_data=reshaped_raw_train,
        batch_size=batch_size,
        val_split=val_split,
        drop=drop_A,
        train_transforms_mock=train_transforms_mock,
        val_transforms_mock=val_transforms_mock,
    )
    train_data_b, val_data_b = dmpair.get_B_data()
    _test_get_x_data(
        train_data=train_data_b,
        val_data=val_data_b,
        raw_data=reshaped_raw_train,
        batch_size=batch_size,
        val_split=val_split,
        drop=drop_B,
        train_transforms_mock=train_transforms_mock,
        val_transforms_mock=val_transforms_mock,
    )
