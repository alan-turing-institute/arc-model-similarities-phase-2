import logging

import numpy as np
from sklearn.neighbors import KernelDensity

from modsim2.similarity.embeddings import EMBEDDING_FN_DICT

from .metrics import DistanceMetric

# Set module logger
logger = logging.getLogger(__name__)


class KDE(DistanceMetric):
    """
    Class to calculate the distance between two datasets
    using kernel density estimation (KDE).
    """

    def __init__(self, seed: int) -> None:
        super().__init__(seed)

    def _pre_process_data(
        self,
        data_A: np.ndarray,
        data_B: np.ndarray,
        embedding_name: str,
    ):
        """
        To do
        """
        # Extract embedding callable
        embedding_fn = EMBEDDING_FN_DICT[embedding_name]

        # In the future, consider optionally using feature embeddings here
        matrix_A = embedding_fn(data_A)
        matrix_B = embedding_fn(data_B)

        return matrix_A, matrix_B

    def calculate_distance(
        self,
        data_A: np.ndarray,
        data_B: np.ndarray,
        labels_A: np.ndarray,
        labels_B: np.ndarray,
        embedding_name: str,
        kernel_name: str,
        # bandwith_values: list,
        metric_name: str,
    ) -> float:
        """
        To do
        """
        # Check for valid embedding choice
        assert embedding_name in EMBEDDING_FN_DICT, "Error: embedding does not exist"

        # Pre-process the data
        processed_A, processed_B = self._pre_process_data(
            data_A=data_A, data_B=data_B, embedding_name=embedding_name
        )

        # This variable is for development / testing only
        num_features = 1

        # Create kernel density estimators for A & B
        estimator_A = KernelDensity(kernel=kernel_name, bandwidth="scott")
        estimator_A.fit(processed_A[:, 0:num_features])
        estimator_B = KernelDensity(kernel=kernel_name, bandwidth="scott")
        estimator_B.fit(processed_B[:, 0:num_features])

        # Generate sample data in the feature space
        samples = np.random.rand(10000, num_features)
        # Score the samples (this generates the log-likelihood)
        score_A = estimator_A.score_samples(samples)
        pdf_values_A = np.exp(score_A)
        score_B = estimator_B.score_samples(samples)
        pdf_values_B = np.exp(score_B)

        # Compute metric
        if metric_name == "l2":
            distance = np.sum(np.power((pdf_values_A - pdf_values_B), 2))
            distance = np.sqrt(distance)
        else:
            raise ValueError("Metric name is not valid:" + metric_name)

        return distance
