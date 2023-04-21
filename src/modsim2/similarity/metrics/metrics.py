from abc import ABC, abstractmethod

import numpy as np
from pytorch_lightning.utilities.seed import seed_everything


class DistanceMetric(ABC):
    def __init__(self, metric_seed: int = 42):
        seed_everything(metric_seed)

    @abstractmethod
    def _pre_process_data(self, data_A: np.ndarray, data_B: np.ndarray):
        """
        This method will take the source and target data
        and will process it into a format so that the distance
        can be calculated. The method will return the processed
        data.

        The steps for processing the data will vary by subclass
        as will the format in which the data are returned.
        """
        pass

    @abstractmethod
    def calculate_distance(
        self,
        data_A: np.ndarray,
        data_B: np.ndarray,
        labels_A: np.ndarray,
        labels_B: np.ndarray,
    ) -> float:
        """
        This method takes the source and target data and labels
        as numpy arrays and returns the distance metric.
        """
        pass
