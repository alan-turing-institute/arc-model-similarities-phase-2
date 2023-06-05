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
    using kernel density estimation (KDE). The density estimation
    is calculated for the source and target (A and B) datasets.
    A metric is then applied to calculate a distance between the
    density distributions.

    sklearn's KernelDensity class has been used as a variety of
    kernels are implemented and the bandwidth can be estimated
    using Scott's Rule.
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
        Takes A & B datasets and an embedding name. Uses the embedding
        to transform the datasets and returns a matrix form of the
        datasets

        Args:
            data_A: the records in the A dataset (excludes target values)
            data_B: the records in the B dataset (excludes target values)
            embedding_name: The feature embedding to be used

        Returns:
            matrix_A: the embedded form of dataset A
            matrix_B: the embedded form of dataset B
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
        metric_name: str,
    ) -> float:
        """
        Calculates the distance between datasets A and B using kernel
        density estimation to represent the probability distribution
        of the datasets.

        Currently only the L2 norm is used to calculate the distance
        between the two datasets - other metrics to follow.

        Args:
            data_A: The first image dataset
            data_B: The second image dataset
            labels_A: The target values for the first dataset
            embedding_name: The feature embeddings to use for the input arrays
            kernel_name: The kernel to be applied in the density estimation, any
                    kernels allowed in sklearns KernelDensity class are valid.
                    Currently these are 'gaussian', 'tophat', 'epanechnikov',
                    'exponential', 'linear', 'cosine'
            metric_name: The metric to be used to calculate the distance between
                    the density functions, currently only 'l2' is valid
        """
        # Check for valid embedding choice
        assert embedding_name in EMBEDDING_FN_DICT, "Error: embedding does not exist"

        # Pre-process the data
        processed_A, processed_B = self._pre_process_data(
            data_A=data_A, data_B=data_B, embedding_name=embedding_name
        )

        # Create kernel density estimators for A & B
        # Scott's rule estimates an appropriate value for the bandwidth, an alternative
        # would be to search bandwidth values using CV
        estimator_A = KernelDensity(kernel=kernel_name, bandwidth="scott")
        estimator_A.fit(processed_A)
        estimator_B = KernelDensity(kernel=kernel_name, bandwidth="scott")
        estimator_B.fit(processed_B)

        # Confirm the two processed datasets have the same number of features
        assert (
            processed_A.shape[1] == processed_B.shape[1]
        ), "Error: datasets A & B do not have the same number of features"
        num_features = processed_A.shape[1]

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
