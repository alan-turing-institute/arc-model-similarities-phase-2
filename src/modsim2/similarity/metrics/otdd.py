import logging
import platform

import numpy as np
import torch
from otdd.pytorch.distance import DatasetDistance
from torch.utils.data import TensorDataset

from .metrics import DistanceMetric

# Set module logger
logger = logging.getLogger(__name__)


class OTDD(DistanceMetric):
    def __init__(self, seed: int = 42):
        super().__init__(seed)

    def _pre_process_data(
        self,
        data_A: np.ndarray,
        data_B: np.ndarray,
        labels_A: np.ndarray,
        labels_B: np.ndarray,
    ):
        # Create a TensorDataset object for A and B consisting of the data and
        # labels
        dataset_A = TensorDataset(torch.tensor(data_A), torch.tensor(labels_A))
        dataset_B = TensorDataset(torch.tensor(data_B), torch.tensor(labels_B))
        # The classes attribute is used when calculating the otdd, so make sure
        # it is defined
        dataset_A.classes = torch.sort(torch.unique(torch.tensor(labels_A))).values
        dataset_B.classes = torch.sort(torch.unique(torch.tensor(labels_B))).values

        return dataset_A, dataset_B

    def calculate_distance(
        self,
        data_A: np.ndarray,
        data_B: np.ndarray,
        labels_A: np.ndarray,
        labels_B: np.ndarray,
        max_samples: int,
        device: str,
        **kwargs,
    ) -> float:
        """
        Calculates the optimal transport distance between datasets

        Args:
            array_A: The first image dataset
            array_B: The second image dataset
            max_samples (int):  maximum number of samples used in outer-level
                        otdd problem.
            device (str): the device on which the calculation will run,
                        e.g. 'cpu', 'mps'

        Returns:
            float: The otdd between A and B
        """
        # the following exception is a placeholder to ensure that code will run on
        # MacBook with M1 processor - is likely to require changing when issue 45 is
        # addressed
        if platform.processor() == "arm" and device == "cpu":
            raise ValueError(
                "The otdd calculation will not produce the correct output "
                "when run on an Apple M1 processor"
            )

        dataset_A, dataset_B = self._pre_process_data(
            data_A, data_B, labels_A, labels_B
        )

        dist = DatasetDistance(dataset_A, dataset_B, device=device, **kwargs)

        # Compute otdd
        logging.info("Computing OTDD. This may take some time!")
        d = dist.distance(maxsamples=max_samples)
        return float(d)
