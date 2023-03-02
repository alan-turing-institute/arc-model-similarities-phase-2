import copy
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from pl_bolts.datamodules import CIFAR10DataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset

from modsim2.similarity.constants import ARGUMENTS, FUNCTION, METRIC_FN_DICT

# Set module logger
logger = logging.getLogger(__name__)


class CIFAR10DMSubset(CIFAR10DataModule):
    def __init__(
        self,
        dataset_train: Union[Subset, Dataset],
        dataset_val: Union[Subset, Dataset],
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
            dataset_train: Dataset or Subset class object to replace
                           original dataset_val with
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

        # Set the datasets directly - change from original CIFAR10 DM
        self.dataset_train = dataset_train
        self.dataset_val = dataset_train
        # need to be able to set transforms on this dataset (underlying the subset)
        #  independently of any other subset so create a shallow copy
        self.dataset_train.dataset = copy.copy(self.dataset_train.dataset)
        self.dataset_val.dataset = copy.copy(self.dataset_val.dataset)

    # Override original setup message to avoid restoring CIFAR observations
    def setup(self, stage: Optional[str] = None) -> None:
        """
        Overrides original setup method in VisionDataModule to avoid reverting
        to original training dataset.

        Args:
            stage: Stage of the training loop. Defaults to None.
        """
        if stage == "fit" or stage is None:
            train_transforms = (
                self.default_transforms()
                if self.train_transforms is None
                else self.train_transforms
            )
            val_transforms = (
                self.default_transforms()
                if self.val_transforms is None
                else self.val_transforms
            )

            self.dataset_train.dataset.transform = train_transforms
            self.dataset_val.dataset.transform = val_transforms

        if stage == "test" or stage is None:
            test_transforms = (
                self.default_transforms()
                if self.test_transforms is None
                else self.test_transforms
            )
            self.dataset_test = self.dataset_cls(
                self.data_dir, train=False, transform=test_transforms, **self.EXTRA_ARGS
            )

    def prepare_data(self):
        # override parent and do nothing
        pass


def split_indices(
    indices: list[int],
    labels: list[int],
    drop_percent_A: float,
    drop_percent_B: float,
    seed: int,
    cifar: CIFAR10DataModule,
) -> Tuple[List[int], List[int]]:
    """
    A function that takes as input a list of indices and a list of
    labels. It returns two lists of indices, A and B, that have dropped
    indices with respect to the input. Importantly however, the indices
    they drop are non-overlapping with respect to each other, and the label
    proportions in A and B are the same as the input, while allowing for
    some rounding error.

    Note that late loading of the train_dataset does not occur.

    Args:
        indices: Indices to be split across 2 datasets
        labels: Labels corresponding to the indices
        drop_percent_A: Percentage of data to drop from A
        drop_percent_B: Percentage of data to drop from B
        seed: Seed for random splitting
        cifar: CIFAR datamodule to use in stratifying the
               second split according to labels

    Returns: Two lists containing indices for A and B
    """

    # Input checking A and B
    sum_drop_percentage = drop_percent_A + drop_percent_B
    if drop_percent_A < 0 or drop_percent_A > 1:
        raise ValueError(
            "drop_percent_A: "
            + str(drop_percent_A)
            + " should be between 0 and 1 inclusive"
        )
    if drop_percent_B < 0 or drop_percent_B > 1:
        raise ValueError(
            "drop_percent_B: "
            + str(drop_percent_B)
            + " should be between 0 and 1 inclusive"
        )
    if sum_drop_percentage > 1:
        raise ValueError(
            "drop_percent_A + drop_percent_B = "
            + str(sum_drop_percentage)
            + " should not be a sum greater than 1"
        )

    # Defaults
    shared_AB_indices = indices
    shared_AB_labels = labels
    indices_kept_A_dropped_B = []
    indices_kept_B_dropped_A = []

    # If needing to drop observations
    if sum_drop_percentage > 0:

        # Numbers to be used
        num_obs = len(indices)
        num_drop_A = round(num_obs * drop_percent_A)
        num_drop_B = round(num_obs * drop_percent_B)
        total_dropped = num_drop_A + num_drop_B
        share_dropped_from_A_kept_in_B = num_drop_A / total_dropped

        # If there will be shared observations between A and B
        if sum_drop_percentage < 1:

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

        # If no shared observations between A and B
        if sum_drop_percentage == 1:
            indices_kept_A_dropped_B, indices_kept_B_dropped_A = train_test_split(
                shared_AB_indices,
                test_size=num_drop_A,
                stratify=shared_AB_labels,
                random_state=seed,
            )
            shared_AB_indices = []

    # Return
    indices_A = shared_AB_indices + indices_kept_A_dropped_B
    indices_B = shared_AB_indices + indices_kept_B_dropped_A
    return indices_A, indices_B


class DMPair:
    def __init__(
        self,
        metric_config: Dict = {},
        drop_percent_A: Union[int, float] = 0,
        drop_percent_B: Union[int, float] = 0,
        transforms_A: Optional[Callable] = None,
        transforms_B: Optional[Callable] = None,
        transforms_test: Optional[Callable] = None,
        data_dir: Optional[str] = None,
        val_split: Union[int, float] = 0.2,
        num_workers: int = 0,
        normalize: bool = False,
        batch_size: int = 32,
        seed: int = 42,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:

        """
        A class to generate and manage two paired datamodules, including
        specifying non-overlapping portions of the original dataset to be dropped.

        Early loading of the train_dataset is performed

        Args:
            metric_config: Dict of metric configs for similarity measures
            drop_percent_A: % of training/val data to drop from A
            drop_percent_B: % of training/val data to drop from B
            transforms_A: transformations applied to A train and val
            transforms_B: transformations applied to B train and val
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
        """

        # Assign params
        self.drop_percent_A = drop_percent_A
        self.drop_percent_B = drop_percent_B
        self.seed = seed
        self.metric_config = metric_config

        # Load and setup CIFAR
        # note: will set transforms later, in CIFAR10DMSubset setup() for A,B
        self.cifar = CIFAR10DataModule(val_split=val_split, seed=self.seed)
        self.cifar.prepare_data()
        logging.warning("Performing early loading of CIFARDM10Subset.dataset_train")
        self.cifar.setup()

        train_indices_A, train_indices_B = self._split_indices(
            dataset=self.cifar.dataset_train
        )
        val_indices_A, val_indices_B = self._split_indices(
            dataset=self.cifar.dataset_val
        )

        # Create data modules
        # NB A and B MUST use the same seed as each other and as the cifar used
        # to generate their training datasets

        self.A = CIFAR10DMSubset(
            dataset_train=Subset(self.cifar.dataset_train.dataset, train_indices_A),
            dataset_val=Subset(self.cifar.dataset_val.dataset, val_indices_A),
            data_dir=data_dir,
            val_split=val_split,
            num_workers=num_workers,
            normalize=normalize,
            batch_size=batch_size,
            seed=seed,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last,
            train_transforms=transforms_A,
            val_transforms=transforms_A,
            test_transforms=transforms_test,
            *args,
            **kwargs,
        )
        self.B = CIFAR10DMSubset(
            dataset_train=Subset(self.cifar.dataset_train.dataset, train_indices_B),
            dataset_val=Subset(self.cifar.dataset_val.dataset, val_indices_B),
            data_dir=data_dir,
            val_split=val_split,
            num_workers=num_workers,
            normalize=normalize,
            batch_size=batch_size,
            seed=seed,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last,
            train_transforms=transforms_B,
            val_transforms=transforms_B,
            test_transforms=transforms_test,
            *args,
            **kwargs,
        )

    def _split_indices(
        self, dataset: Union[Subset, Dataset]
    ) -> Tuple[List[int], List[int]]:
        # if empty then return two empty lists
        if len(dataset) == 0:
            return [], []
        # else
        return split_indices(
            indices=dataset.indices,
            labels=[image[1] for image in dataset],
            drop_percent_A=self.drop_percent_A,
            drop_percent_B=self.drop_percent_B,
            seed=self.seed,
            cifar=self.cifar,
        )

    def compute_similarity(self, only_train: bool = False):
        """
        compute similarity between data of A and B
        only_train removes the validation data from this comparison
        """

        # coerce data into single tensor (not subset)
        # TODO issue-15 want to get data post-transform
        train_data_A, val_data_A = self.get_A_data()
        train_data_B, val_data_B = self.get_B_data()

        if not only_train:
            data_A = np.concatenate((train_data_A, val_data_A), axis=0)
            data_B = np.concatenate((train_data_B, val_data_B), axis=0)
        else:
            data_A = train_data_A
            data_B = train_data_B

        # Loop over dict, compute metrics
        similarity_dict = {}
        for key, metric in self.metric_config.items():
            similarity_dict[key] = METRIC_FN_DICT[metric[FUNCTION]](
                data_A, data_B, **metric[ARGUMENTS]
            )

        # Output
        return similarity_dict

    """ convenience methods below for getting data/labels from subsets """

    @staticmethod
    def _get_subset_data(
        subset_module: CIFAR10DMSubset,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # force __getitem__ in CIFAR10 class for train and val
        # just take image rather than label
        train = torch.stack([item[0] for item in subset_module.dataset_train])
        val = torch.stack([item[0] for item in subset_module.dataset_val])
        return train, val

    def get_A_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._get_subset_data(self.A)

    def get_B_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._get_subset_data(self.B)

    @staticmethod
    def _get_subset_labels(subset_module: CIFAR10DMSubset) -> Tuple[List, List]:
        # List comprehension because pytorch dataset makes it necsesary
        # don't need to apply transforms here so don't bother forcing __getitem__
        train = [
            subset_module.dataset_train.dataset.targets[i]
            for i in subset_module.dataset_train.indices
        ]
        val = [
            subset_module.dataset_val.dataset.targets[i]
            for i in subset_module.dataset_val.indices
        ]
        return train, val

    def get_A_labels(self) -> Tuple[List, List]:
        """returns train and val labels for A"""
        return self._get_subset_labels(self.A)

    def get_B_labels(self) -> Tuple[List, List]:
        """returns train and val labels for A"""
        return self._get_subset_labels(self.B)
