from modsim2.data.loader import CIFAR10DataModuleDrop


def testDropLoader():
    cifar = CIFAR10DataModuleDrop(drop=0.1)
    cifar.prepare_data()
    cifar.setup()
    dl = cifar.train_dataloader()
    assert len(dl.dataset) == 36000
