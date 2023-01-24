from typing import Any, Callable, Optional, Union

from pl_bolts.datamodules import CIFAR10DataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset


class CIFAR10DMSubset(CIFAR10DataModule):
    def __init__(
        self,
        dataset_train: Union[Subset, Dataset],
        data_dir: Optional[str] = None,
        val_split: Union[int, float] = 0.2,
        num_workers: int = 0,
        normalize: bool = False,
        batch_size: int = 32,
        seed: int = 42,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        train_transforms: Optional[Callable] = None,
        val_transforms: Optional[Callable] = None,
        test_transforms: Optional[Callable] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        A modified CIFAR10DataModule class that takes a Subset as input and replaces
        the original training dataset with this Subset

        Args:
            dataset_train: Dataset or Subset class object to replace
                           original dataset_train with
            data_dir: Where to save/load the data
            val_split: Percent (float) or number (int) of samples to use
                       for the validation split
            num_workers: How many workers to use for loading data
            normalize: If true applies image normalize
            batch_size: How many samples per batch to load
            seed: Random seed to be used for train/val/test splits
            shuffle: If true shuffles the train data every epoch
            pin_memory: If true, the data loader will copy Tensors into
                        CUDA pinned memory before
                        returning them
            drop_last: If true drops the last incomplete batch
            train_transforms: transformations you can apply to train dataset
            val_transforms: transformations you can apply to validation dataset
            test_transforms: transformations you can apply to test dataset
        """

        # Call the super
        super().__init__(
            data_dir=data_dir,
            val_split=val_split,
            num_workers=num_workers,
            normalize=normalize,
            batch_size=batch_size,
            seed=seed,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            *args,
            **kwargs,
        )

        # Change the dataset from original CIFAR10 DM
        self.dataset_train = dataset_train


def split_indices(
    indices: list[int],
    labels: list[int],
    drop_percent_A: float,
    drop_percent_B: float,
    seed: int,
    cifar: CIFAR10DataModule,
) -> tuple[list[int], list[int]]:
    """_summary_

    Args:
        indices (list[int]): Indices to be split across 2 datasets
        labels (list[int]): Labels corresponding to the indices
        drop_percent_A (float): Percentage of data to drop from A
        drop_percent_B (float): Percentage of data to drop from B
        seed (int): Seed for random splitting
        cifar (CIFAR10DataModule): CIFAR datamodule to use in extracting
                                   stratifying the second split

    Returns:
        tuple[list[int], list[int]]: Two lists containing indices for A and B
    """

    # Defaults
    shared_AB_indices = indices
    shared_AB_labels = labels
    indices_kept_A_dropped_B = []
    indices_kept_B_dropped_A = []

    # Numbers to be used
    num_obs = len(indices)
    num_drop_A = round(num_obs * drop_percent_A)
    num_drop_B = round(num_obs * drop_percent_B)
    total_dropped = num_drop_A + num_drop_B
    share_dropped_from_A_kept_in_B = num_drop_A / total_dropped

    # If needing to drop observations
    if (drop_percent_A + drop_percent_B) > 0:

        # Split (drop_A + drop_B)% of the training data
        shared_AB_indices, drop_indices = train_test_split(
            shared_AB_indices,
            test_size=total_dropped,
            stratify=shared_AB_labels,
            random_state=seed,
        )
        drop_labels = [cifar.dataset_train.dataset.targets[i] for i in drop_indices]

        # If dropping only from A
        if drop_percent_A > 0 and drop_percent_B == 0:
            indices_kept_B_dropped_A = drop_indices

        # If dropping only from B
        if drop_percent_A == 0 and drop_percent_B > 0:
            indices_kept_A_dropped_B = drop_indices

        # If dropping from both
        if (drop_percent_A > 0) and (drop_percent_B > 0):

            # Split the unselected component into parts to keep/drop in A vs B
            # since B is test, test amount is proportion of unselected that is B
            indices_kept_A_dropped_B, indices_kept_B_dropped_A = train_test_split(
                drop_indices,
                test_size=share_dropped_from_A_kept_in_B,
                stratify=drop_labels,
                random_state=seed,
            )

    # Return
    indices_A = shared_AB_indices + indices_kept_A_dropped_B
    indices_B = shared_AB_indices + indices_kept_B_dropped_A
    return indices_A, indices_B


class DMPair:
    def __init__(
        self,
        drop_percent_A: Union[int, float] = 0,
        drop_percent_B: Union[int, float] = 0,
        data_dir: Optional[str] = None,
        val_split: Union[int, float] = 0.2,
        num_workers: int = 0,
        normalize: bool = False,
        batch_size: int = 32,
        seed: int = 42,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        train_transforms: Optional[Callable] = None,
        val_transforms: Optional[Callable] = None,
        test_transforms: Optional[Callable] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:

        """
        A class to generate and manage two paired datamodules, including
        specifying non-overlapping portions of the original dataset to be dropped

        Args:
            drop_percent_A: % of training data to drop from A
            drop_percent_B: % of training data to drop from B
            data_dir: Where to save/load the data
            val_split: Percent (float) or number (int) of samples to use
                       for the validation split
            num_workers: How many workers to use for loading data
            normalize: If true applies image normalize
            batch_size: How many samples per batch to load
            seed: Random seed to be used for train/val/test splits and
                  random dropping of observations
            shuffle: If true shuffles the train data every epoch
            pin_memory: If true, the data loader will copy Tensors into
                        CUDA pinned memory before
                        returning them
            drop_last: If true drops the last incomplete batch
            train_transforms: transformations you can apply to train dataset
            val_transforms: transformations you can apply to validation dataset
            test_transforms: transformations you can apply to test dataset
        """

        # Assign params
        self.drop_percent_A = drop_percent_A
        self.drop_percent_B = drop_percent_B
        self.seed = seed

        # Load and setup CIFAR
        cifar = CIFAR10DataModule(val_split=val_split, seed=self.seed)
        cifar.prepare_data()
        cifar.setup()

        # Default
        shared_AB_indices = cifar.dataset_train.indices
        shared_AB_labels = [image[1] for image in cifar.dataset_train]
        self.indices_A, self.indices_B = split_indices(
            indices=shared_AB_indices,
            labels=shared_AB_labels,
            drop_percent_A=self.drop_percent_A,
            drop_percent_B=self.drop_percent_B,
            seed=self.seed,
            cifar=cifar,
        )

        # Create data modules
        self.cifar = cifar  # necessary for some tests
        self.A = CIFAR10DMSubset(
            dataset_train=Subset(cifar.dataset_train.dataset, self.indices_A),
            data_dir=data_dir,
            val_split=val_split,
            num_workers=num_workers,
            normalize=normalize,
            batch_size=batch_size,
            seed=seed,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            *args,
            **kwargs,
        )
        self.B = CIFAR10DMSubset(
            dataset_train=Subset(cifar.dataset_train.dataset, self.indices_B),
            data_dir=data_dir,
            val_split=val_split,
            num_workers=num_workers,
            normalize=normalize,
            batch_size=batch_size,
            seed=seed,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            *args,
            **kwargs,
        )

        # Store labels
        # List comprehension because pytorch dataset makes it necsesary
        self.labels_A = [
            self.A.dataset_train.dataset.targets[i] for i in self.indices_A
        ]
        self.labels_B = [
            self.B.dataset_train.dataset.targets[i] for i in self.indices_B
        ]
