from typing import Any, Callable, Optional, Union

from pl_bolts.datamodules import CIFAR10DataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset


class CIFAR10DataModulePlus(CIFAR10DataModule):
    def __init__(
        self,
        drop: Union[int, float] = 0,
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

        # The new bit: drop %
        self.drop = drop

    # Allow updating the drop value (for A/B copying then changing)
    def update_drop(self, new_drop: Union[int, float]):
        self.drop = new_drop

    # Data loader: if drop > 0, drop that % of the data
    def train_dataloader(self) -> DataLoader:
        if self.drop > 0:
            labels = [i[1] for i in self.dataset_train]
            index, _ = train_test_split(
                self.dataset_train.indices,
                test_size=self.drop,
                stratify=labels,
                random_state=self.seed,
            )
            return self._data_loader(
                Subset(self.dataset_train, index), shuffle=self.shuffle
            )
        else:
            return self._data_loader(self.dataset_train, shuffle=self.shuffle)
