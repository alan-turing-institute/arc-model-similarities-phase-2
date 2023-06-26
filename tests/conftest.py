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
def randomNormalDifferentInceptionMock():
    """
    produce random samples from normal distr
    ensures each subsequent call has a different mean
    """
    # note, will have a limit on times it can be called (size of range)
    mean_iter = range(MEAN_LOW, MEAN_HIGH).__iter__()

    def random_inception_embeds(
        array: np.ndarray,
        batch_size: int,
        device: str,
    ) -> np.ndarray:
        try:
            loc = mean_iter.__next__()
        except StopIteration:
            raise Exception(
                "too many calls to this mock, only 200 iterations supported"
            )
        return np.random.default_rng().normal(
            loc=loc, scale=SCALE, size=(array.shape[0], 2048)
        )

    mock = Mock(side_effect=random_inception_embeds)
    with patch("modsim2.similarity.embeddings.inception", new=mock):
        yield None


@pytest.fixture(scope="module", autouse=False)
def randomNormalDifferentUmapMock():
    """
    produce random samples from normal distr
    ensures each subsequent call has a different mean
    """
    # note, will have a limit on times it can be called (size of range)
    mean_iter = range(MEAN_LOW, MEAN_HIGH).__iter__()

    def random_umap_embeds(
        array: np.ndarray, random_seed: int, n_components: int
    ) -> np.ndarray:
        try:
            loc = mean_iter.__next__()
        except StopIteration:
            raise Exception(
                "too many calls to this mock, only 200 iterations supported"
            )
        return np.random.default_rng().normal(
            loc=loc, scale=SCALE, size=(array.shape[0], n_components)
        )

    mock = Mock(side_effect=random_umap_embeds)
    with patch("modsim2.similarity.embeddings.umap", new=mock):
        yield None


@pytest.fixture(scope="module", autouse=False)
def randomNormalDifferentPcaMock():
    """
    produce random samples from normal distr
    ensures each subsequent call has a different mean
    """
    # note, will have a limit on times it can be called (size of range)
    mean_iter = range(MEAN_LOW, MEAN_HIGH).__iter__()

    def random_pca_embeds(array: np.ndarray, n_components: int) -> np.ndarray:
        try:
            loc = mean_iter.__next__()
        except StopIteration:
            raise Exception(
                "too many calls to this mock, only 200 iterations supported"
            )
        return np.random.default_rng().normal(
            loc=loc, scale=SCALE, size=(array.shape[0], n_components)
        )

    mock = Mock(side_effect=random_pca_embeds)
    with patch("modsim2.similarity.embeddings.pca", new=mock):
        yield None


# fixed values - e.g. when we want repeated calls to generate same thing


@pytest.fixture(scope="module", autouse=False)
def fixedInceptionMock():
    def fixed_inception_embeds(
        array: np.ndarray,
        batch_size: int,
        device: str,
    ) -> np.ndarray:
        return np.ones(array.shape[0], 2048)

    mock = Mock(side_effect=fixed_inception_embeds)
    with patch("modsim2.similarity.embeddings.inception", new=mock):
        yield None


@pytest.fixture(scope="module", autouse=False)
def fixedUmapMock():
    def fixed_umap_embeds(
        array: np.ndarray, random_seed: int, n_components: int
    ) -> np.ndarray:
        return np.ones(array.shape[0], n_components)

    mock = Mock(side_effect=fixed_umap_embeds)
    with patch("modsim2.similarity.embeddings.umap", new=mock):
        yield None


@pytest.fixture(scope="module", autouse=False)
def fixedPcaMock():
    def fixed_pca_embeds(array: np.ndarray, n_components: int) -> np.ndarray:
        return np.ones(array.shape[0], n_components)

    mock = Mock(side_effect=fixed_pca_embeds)
    with patch("modsim2.similarity.embeddings.pca", new=mock):
        yield None
