from typing import Any, Callable, Optional, Union

from pl_bolts.datamodules import CIFAR10DataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset


class CIFAR10DMSubset(CIFAR10DataModule):
    def __init__(
        self,
        train_dataset: Union[Subset, Dataset],
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
        A modified CIFAR10DataModule class that drops a % of the train set

        Args:
            drop: % of training data to drop when using dataloader
            keep: whether in the 'unselected' component to keep A or B
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
        self.dataset_train = train_dataset


class DMPair:
    def __init__(
        self,
        drop_A: Union[int, float] = 0,
        drop_B: Union[int, float] = 0,
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
    ):
        # Assign params
        self.drop_A = drop_A
        self.drop_B = drop_B
        self.seed = seed

        # Load and setup CIFAR
        cifar = CIFAR10DataModule()
        cifar.prepare_data()
        cifar.setup()

        # If not dropping any observations
        if (self.drop_A + self.drop_B) == 0:
            self.A = cifar
            self.B = cifar

        # If needing to drop observations
        if (self.drop_A + self.drop_B) > 0:

            # Set up indices and labels
            self._train_indices = cifar.dataset_train.indices
            self._train_labels = [i[1] for i in cifar.dataset_train]

            # Defaults
            a_inds = []
            b_inds = []

            # Split (drop_A + drop_B)% of the training data
            self._main_indices, self._drop_indices = train_test_split(
                self._train_indices,
                test_size=self.drop_A + self.drop_B,  # drop all together
                stratify=self._train_labels,
                random_state=self.seed,
            )
            unselected_labels = [
                cifar.dataset_train.dataset.targets[i] for i in self._drop_indices
            ]

            # If dropping only from A
            if self.drop_A > 0 and self.drop_B == 0:
                b_inds = unselected_labels

            # If dropping only from B
            if self.drop_A == 0 and self.drop_B > 0:
                a_inds = unselected_labels

            # If dropping from both
            if (self.drop_A > 0) and (self.drop_B > 0):

                # Split the unselected component into parts to keep/drop in A vs B
                # since B is test, test amount is proportion of unselected that is B
                a_inds, b_inds = train_test_split(
                    self._drop_indices,
                    test_size=self.drop_B / (self.drop_A + self.drop_B),
                    stratify=unselected_labels,
                    random_state=self.seed,
                )

            # Store indices
            self.indices_A = self._main_indices + a_inds
            self.indices_B = self._main_indices + b_inds

            # Create data modules
            self.A = CIFAR10DMSubset(
                train_dataset=Subset(cifar.dataset_train, self.indices_A),
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
                train_dataset=Subset(cifar.dataset_train, self.indices_B),
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
