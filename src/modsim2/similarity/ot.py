import logging

import numpy as np
from otdd.pytorch.distance import DatasetDistance

# Set module logger
logger = logging.getLogger(__name__)


def ot(
    array_A: np.ndarray,
    array_B: np.ndarray,
    implementation: str = "otdd",
    max_samples: int = 10,
    **kwargs,
) -> float:
    """
    Calculates the optimal transport distance between datasets

    Args:
        array_A: The first image dataset
        array_B: The second image dataset
        implementation (str): the implementation type of OT, currently only 'otdd'
                              is accepted
        max_samples (int):  maximum number of samples used in outer-level otdd problem.


    Returns:
        float: The ot between A and B
    """
    if implementation == "otdd":
        dist = DatasetDistance(array_A, array_B, **kwargs)
        d = dist.distance(maxsamples=max_samples)
        return d
    else:
        raise ValueError("Unknown implementation:" + implementation)
