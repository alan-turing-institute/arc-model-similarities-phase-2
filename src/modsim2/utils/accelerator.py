import pytorch_lightning.accelerators


def choose_auto_accelerator() -> str:
    """
    Code adapated from pytorch_lightning AcceleratorConnector class method to detect
    appropriate accelator. Only needed when requiring auto detection
    """
    if pytorch_lightning.accelerators.tpu.TPUAccelerator.is_available():
        return "tpu"
    if pytorch_lightning.accelerators.ipu.IPUAccelerator.is_available():
        return "ipu"
    if pytorch_lightning.accelerators.hpu.HPUAccelerator.is_available():
        return "hpu"
    if pytorch_lightning.accelerators.mps.MPSAccelerator.is_available():
        return "mps"
    if pytorch_lightning.accelerators.cuda.CUDAAccelerator.is_available():
        return "cuda"
    return "cpu"
