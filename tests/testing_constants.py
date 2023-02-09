import os

VAL_SPLIT = 0.2
DUMMY_CIFAR10_SIZE = 300
DUMMY_CIFAR10_TRAIN_SIZE = DUMMY_CIFAR10_SIZE * (1 - VAL_SPLIT)
DUMMY_CIFAR_DIR = os.path.abspath(
    os.path.join(__file__, os.pardir, "testdata", "dummy_cifar")
)
