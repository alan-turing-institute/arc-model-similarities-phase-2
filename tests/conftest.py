from unittest.mock import patch

import pytest
import testing_constants
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.datasets.cifar10_dataset import CIFAR10


# same structure as CIFAR10 but doesn't require a download
class DummyCIFAR10(CIFAR10):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_samples = testing_constants.DUMMY_CIFAR10_SIZE

    @property
    def cached_folder_path(self) -> str:
        return testing_constants.DUMMY_CIFAR_DIR

    def prepare_data(self, download: bool):
        pass


class MockCIFAR10DataModule(CIFAR10DataModule):
    dataset_cls = DummyCIFAR10


@pytest.fixture(scope="module", autouse=True)
def patch_datamodule():
    with patch("modsim2.data.loader.CIFAR10DataModule", new=MockCIFAR10DataModule):
        yield None
