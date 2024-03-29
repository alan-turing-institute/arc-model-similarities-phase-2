import logging

import numpy as np
from scipy import integrate
from sklearn.neighbors import KernelDensity, NearestNeighbors

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
        embedding_kwargs: dict,
    ) -> tuple[np.ndarray, np.ndarray]:

        # Embed the data
        embed_A, embed_B = self._embed_data(
            data_A=data_A,
            data_B=data_B,
            embedding_name=embedding_name,
            embedding_kwargs=embedding_kwargs,
        )

        return embed_A, embed_B

    @staticmethod
    def _kde_distance(
        num_dimensions: int,
        func: callable,
        integration_kwargs: dict,
    ) -> tuple[float, float]:
        """
        Integrates a given function over a number of dimensions. Returns the result
        of the integration

        Args:
            num_dimensions : the number of dimensions that are to be integrated over
            func: the function to be integrated
            integration_kwargs: A dictionary of key word arguments to be passed to the
                            integration function

        Returns:
            distance: the result of the integration
            abs_error: the maximum of the estimates of the absolute error in the
                    various integration results.
        """

        # the bounds over which the integration will be performed
        bounds = [[-np.inf, np.inf] for _ in range(num_dimensions)]

        # perform integration, returns distance and absolute error
        distance, abs_error = integrate.nquad(func, bounds, **integration_kwargs)

        logger.warn("Absolute error of integration: %s", abs_error)

        return distance

    @staticmethod
    def l2(
        num_dimensions: int,
        estimator_A: KernelDensity,
        estimator_B: KernelDensity,
        integration_kwargs: dict,
    ) -> tuple[float, float]:
        """
        Calculates the L2 Norm between two probability density estimatiors for a given
        number of dimensions using integration

        Args:
            num_dimensions : the number of dimensions that are to be integrated over
            estimator_A: the density estimator that has been generated for a dataset A
            estimator_B: the density estimator that has been generated for a dataset B
            integration_kwargs: A dictionary of key word arguments to be passed to the
                            integration function

        Returns:
            distance: the L2 distance
            abs_error: the absolute error of the integration
        """
        # the function to be integrated
        def func(*args):
            # log-likelihood of estimators A & B
            score_A = estimator_A.score([[*args]])
            score_B = estimator_B.score([[*args]])
            # apply the exp function to return the pdf
            pdf_A = np.exp(score_A)
            pdf_B = np.exp(score_B)
            # l2 norm between the two pdfs - difference raised to power of 2
            l2_func = np.power(pdf_A - pdf_B, 2)
            return l2_func

        # Perform integration
        distance = KDE._kde_distance(
            num_dimensions=num_dimensions,
            func=func,
            integration_kwargs=integration_kwargs,
        )
        distance = np.sqrt(distance)

        return distance, distance

    @staticmethod
    def total_variation(
        num_dimensions: int,
        estimator_A: KernelDensity,
        estimator_B: KernelDensity,
        integration_kwargs: dict,
    ) -> tuple[float, float]:
        """
        Calculates the total variation between two probability density estimatiors for
        a given number of dimensions using integration

        Args:
            num_dimensions : the number of dimensions that are to be integrated over
            estimator_A: the density estimator that has been generated for a dataset A
            estimator_B: the density estimator that has been generated for a dataset B
            integration_kwargs: A dictionary of key word arguments to be passed to the
                            integration function

        Returns:
            distance: the total variation distance
            abs_error: the absolute error of the integration
        """
        # the function to be integrated
        def func(*args):
            # log-likelihood of estimators A & B
            score_A = estimator_A.score([[*args]])
            score_B = estimator_B.score([[*args]])
            # apply the exp function to return the pdf
            pdf_A = np.exp(score_A)
            pdf_B = np.exp(score_B)
            # total variation between the two pdfs - half the absolute difference
            tv_func = 0.5 * np.abs(pdf_A - pdf_B)
            return tv_func

        # perform integration
        distance = KDE._kde_distance(
            num_dimensions=num_dimensions,
            func=func,
            integration_kwargs=integration_kwargs,
        )

        return distance, distance

    _kde_metric_dict = {
        "l2": l2,
        "total_variation": total_variation,
    }

    @staticmethod
    def kl_tree(A: np.ndarray, B: np.ndarray):
        """
        Calculates the KL divergence between two datasets for a given number of
        using an approximate method with a nearest neighbour algorithm detailed
        in this paper:
        https://ieeexplore.ieee.org/document/4595271

        Note, there is a mistake in equation 14 of the paper which defines the
        divergence as:
        divergence = np.log(r/s).sum() * d/n + np.log(m / (n-1))

        The divergence should either be:
        divergence = np.log(s/r).sum() * d/n + np.log(m / (n-1))
        or:
        divergence = -np.log(r/s).sum() * d/n + np.log(m / (n-1))

        The KL divergence should always be positive. However, for small values
        of m and n, it is possible that this method will calculate a negative
        value. The referenced paper proves that the equation converges almost
        surely to the KL divergence as m, n -> inf

        Args:
            A: dataset A
            B: dataset B

        Returns:
            divergence: the approximate KL divergence
        """
        assert (
            A.shape[1] == B.shape[1]
        ), "Error: A & B must have the same number of features"

        n, d = A.shape
        m, _ = B.shape

        # Fit nearest neighbour estimators using datasets A & B
        # For the A estimator we need two nearest neighbours as we will be comparing A
        # to A (and the nearest neighbour to each sample will be itself)
        # For the B estimator we only need one nearest neighbour as we will comparing A
        # to B
        A_neighbourhood = NearestNeighbors(n_neighbors=2, algorithm="kd_tree").fit(A)
        B_neighbourhood = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(B)

        # Find the distances between samples in A and A
        AA_distances, _ = A_neighbourhood.kneighbors(A, 2)
        # Find the distances between samples in A and B
        AB_distances, _ = B_neighbourhood.kneighbors(A, 1)

        # Take the last distance for each
        r = AA_distances[:, -1]
        s = AB_distances[:, -1]

        # Calculate distance according to corrected eqn 14 in paper
        distance = np.log(s / r).sum() * d / n + np.log(m / (n - 1))

        return distance

    def calculate_distance(
        self,
        data_A: np.ndarray,
        data_B: np.ndarray,
        labels_A: np.ndarray,
        labels_B: np.ndarray,
        embedding_name: str,
        kernel_name: str,
        metric_name: str,
        embedding_kwargs: dict = {},
        integration_kwargs: dict = {},
    ) -> tuple[float, float]:
        """
        Calculates the distance between datasets A and B using kernel
        density estimation to represent the probability density function
        of the datasets.

        The L2 norm and total variation are options that are provided
        to calculate the distance between the distributions. The KL
        divergence between the distributions can also be calculated;
        however, this is not a true metric as it is not symmetrical.
        An approximate method for calculating the KL divergence is
        provided, although this does not require the density estimation.

        Args:
            data_A: The first image dataset
            data_B: The second image dataset
            labels_A: The target values for the first dataset
            labels_B: The target values for the second dataset
            embedding_name: The feature embeddings to use for the input arrays
            kernel_name: The kernel to be applied in the density estimation, any
                    kernels allowed in sklearns KernelDensity class are valid.
                    Currently these are 'gaussian', 'tophat', 'epanechnikov',
                    'exponential', 'linear', 'cosine'
            metric_name: The metric to be used to calculate the distance between
                    the density functions, currently only 'l2' is valid
            embedding_kwargs: A dictionary of key word arguments to be passed
                    to the embedding function
            integration_kwargs: A dictionary of key word arguments to be passed
                    to the integration function
        """
        # A value of zero is returned if the datasets are the same (the integration
        # methods cannot be used if the datasets are the same, and the approximate
        # kl method will calculate as -inf)
        if np.array_equal(data_A, data_B):
            return 0, 0

        # Embed the data
        embed_A, embed_B = self._pre_process_data(
            data_A=data_A,
            data_B=data_B,
            embedding_name=embedding_name,
            embedding_kwargs=embedding_kwargs,
        )

        if metric_name == "kl_approx":
            # The kernel density estimation is not required
            distance_AB = self.kl_tree(embed_A, embed_B)
            distance_BA = self.kl_tree(embed_B, embed_A)
            return distance_AB, distance_BA

        else:
            # Create kernel density estimators for A & B
            # Scott's rule estimates an appropriate value for the bandwidth, an
            # alternative would be to search bandwidth values using CV
            estimator_A = KernelDensity(kernel=kernel_name, bandwidth="scott")
            estimator_A.fit(embed_A)
            estimator_B = KernelDensity(kernel=kernel_name, bandwidth="scott")
            estimator_B.fit(embed_B)

            # Confirm the two embedded datasets have the same number of features
            # before setting the number of features
            if embed_A.shape[1] != embed_B.shape[1]:
                raise ValueError(
                    "Datasets A & B do not have the same number of features"
                )
            num_features = embed_A.shape[1]

            # Compute and return distance
            return self._kde_metric_dict[metric_name](
                num_features, estimator_A, estimator_B, integration_kwargs
            )
