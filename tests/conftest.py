import logging
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


# used for all tests - need to revisit mock deletion failure if this changes
@pytest.fixture(scope="module", autouse=True)
def patch_datamodule():
    with patch("modsim2.data.loader.CIFAR10DataModule", new=MockCIFAR10DataModule):
        # patch for inheritance within CIFAR10DMSubset
        # alternative is to monkey patch in each test
        # annoyingly need to catch exception on exit of mock as
        #  it messes up removing itself
        try:
            with patch(
                "modsim2.data.loader.CIFAR10DMSubset.__bases__",
                (MockCIFAR10DataModule,),
            ):
                yield None
        except TypeError as e:
            # make sure it's the error we're expected and just pass if so
            # should be fine as used for all tests
            if (
                str(e)
                == "cannot delete '__bases__' attribute of immutable type 'CIFAR10DMSubset'"  # noqa: E501
            ):
                # warn
                logging.warning("Failed to delete mock on CIFAR10DMSubset inheritace.")
                pass
            else:
                raise e
