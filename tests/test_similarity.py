import os
from unittest.mock import patch

import pytest
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.datasets.cifar10_dataset import CIFAR10

from modsim2.data.loader import DMPair

# Constants for testing
VAL_SPLIT = 0.2
DUMMY_CIFAR10_SIZE = 300
DUMMY_CIFAR10_TRAIN_SIZE = DUMMY_CIFAR10_SIZE * (1 - VAL_SPLIT)
DUMMY_CIFAR_DIR = os.path.abspath(
    os.path.join(__file__, os.pardir, "testdata", "dummy_cifar")
)


def _test_dm_n_obs(
    length_subset: int,
    drop: float,
) -> None:
    assert length_subset == (1 - drop) * DUMMY_CIFAR10_TRAIN_SIZE


# same structure as CIFAR10 but doesn't require a download
class DummyCIFAR10(CIFAR10):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_samples = DUMMY_CIFAR10_SIZE

    @property
    def cached_folder_path(self) -> str:
        return DUMMY_CIFAR_DIR

    def prepare_data(self, download: bool):
        pass


class MockCIFAR10DataModule(CIFAR10DataModule):
    dataset_cls = DummyCIFAR10


@pytest.fixture(scope="module", autouse=True)
def patch_datamodule():
    with patch("modsim2.data.loader.CIFAR10DataModule", new=MockCIFAR10DataModule):
        yield None


def test_cifar_mmd_same():
    dmpair = DMPair()
    similarity_dict = dmpair.compute_similarity()
    assert similarity_dict["mmd"] == 0


def test_cifar_mmd_different():
    dmpair = DMPair(drop_percent_A=0.2)
    similarity_dict = dmpair.compute_similarity()
    assert similarity_dict["mmd"] > 0
