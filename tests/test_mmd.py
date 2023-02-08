import os
from unittest.mock import patch

import pytest
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.datasets.cifar10_dataset import CIFAR10

from modsim2.data.loader import DMPair
from modsim2.utils.config import load_configs

# Constants for testing
VAL_SPLIT = 0.2
DUMMY_CIFAR10_SIZE = 300
DUMMY_CIFAR10_TRAIN_SIZE = DUMMY_CIFAR10_SIZE * (1 - VAL_SPLIT)
DUMMY_CIFAR_DIR = os.path.abspath(
    os.path.join(__file__, os.pardir, "testdata", "dummy_cifar")
)

# project root = arc-model-similarites-phase-2/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Get metric config
METRICS_CONFIG_PATH = os.path.join(PROJECT_ROOT, "tests", "testconfig", "metrics.yaml")
CONFIGS = load_configs(METRICS_CONFIG_PATH)
METRIC_CONFIG = CONFIGS["metric_config"]


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
    dmpair = DMPair(metric_config=METRIC_CONFIG)
    similarity_dict = dmpair.compute_similarity()
    assert similarity_dict["mmd_rbf"] == 0
    assert similarity_dict["mmd_laplace"] == 0


def test_cifar_mmd_different():
    dmpair = DMPair(metric_config=METRIC_CONFIG, drop_percent_A=0.2)
    similarity_dict = dmpair.compute_similarity()
    assert similarity_dict["mmd_rbf"] > 0
    assert similarity_dict["mmd_laplace"] > 0
