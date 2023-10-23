import os

DUMMY_CIFAR10_SIZE = 300
DUMMY_CIFAR10_TEST_SIZE = 100
DUMMY_CIFAR_DIR = os.path.abspath(
    os.path.join(__file__, os.pardir, "testdata", "dummy_cifar")
)
DUMMY_CIFAR_TRAIN = os.path.join(DUMMY_CIFAR_DIR, "training.pt")
DUMMY_CIFAR_TEST = os.path.join(DUMMY_CIFAR_DIR, "test.pt")
