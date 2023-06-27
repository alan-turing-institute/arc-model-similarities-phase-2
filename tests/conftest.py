import logging
from unittest.mock import Mock, patch

import numpy as np
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


###
# embedding mocks
###

# random values - repeated calls give different vals

# consts for normal distr
MEAN_LOW = -100
MEAN_HIGH = 100
SCALE = 0.1


@pytest.fixture(scope="module", autouse=False)
def inceptionMock():
    """
    produce random samples from normal distr
    ensures each subsequent call has a different mean
    """

    def inception_embeds(
        array: np.ndarray,
        batch_size: int,
        device: str,
    ) -> np.ndarray:
        # assume feature unrolled is larger than 2048
        assert np.prod(array.shape[1:]) > 2048
        # just return data itsself (also in range 0,1)
        return array.reshape(array.shape[0], -1)[:, :2048]

    mock = Mock(side_effect=inception_embeds)
    with patch("modsim2.similarity.embeddings.inception", new=mock):
        yield None


@pytest.fixture(scope="module", autouse=False)
def umapMock():
    def umap_embeds(
        array: np.ndarray, random_seed: int, n_components: int
    ) -> np.ndarray:
        # assume feature unrolled is larger than n_components
        assert np.prod(array.shape[1:]) > n_components
        # just return data itsself (also in range 0,1)
        return array.reshape(array.shape[0], -1)[:, :n_components]

    mock = Mock(side_effect=umap_embeds)
    with patch("modsim2.similarity.embeddings.umap", new=mock):
        yield None


@pytest.fixture(scope="module", autouse=False)
def pcaMock():
    def pca_embeds(array: np.ndarray, n_components: int) -> np.ndarray:
        # assume feature unrolled is larger than n_components
        assert np.prod(array.shape[1:]) > n_components
        # just return data itsself (also in range 0,1)
        return array.reshape(array.shape[0], -1)[:, :n_components]

    mock = Mock(side_effect=pca_embeds)
    with patch("modsim2.similarity.embeddings.pca", new=mock):
        yield None
