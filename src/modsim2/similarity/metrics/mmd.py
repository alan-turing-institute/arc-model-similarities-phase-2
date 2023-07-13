import logging

import numpy as np
from sklearn import metrics

from .metrics import DistanceMetric

# Set module logger
logger = logging.getLogger(__name__)


class MMD(DistanceMetric):
    def __init__(self, seed: int):
        super().__init__(seed)
        # Kernel dictionary
        self.__mmd_kernel_dict = {
            "rbf": metrics.pairwise.rbf_kernel,
            "laplace": metrics.pairwise.laplacian_kernel,
        }

    def _pre_process_data(
        self,
        data_A: np.ndarray,
        data_B: np.ndarray,
        embedding_name: str,
        kernel_name: str,
        embedding_kwargs: dict,
    ):
        matrix_A, matrix_B = self._embed_data(
            data_A, data_B, embedding_name, embedding_kwargs
        )

        N_A = matrix_A.shape[0]
        N_B = matrix_B.shape[0]

        # Extract kernel callable
        kernel_fn = self.__mmd_kernel_dict[kernel_name]

        # Compute MMD components
        logging.info(
            f"Computing {N_A} by {N_A}, {N_B} by {N_B}, and {N_A} by {N_B} kernels. "
            "This may take some time!"
        )
        kernel_AA = kernel_fn(X=matrix_A)
        kernel_BB = kernel_fn(X=matrix_B)
        kernel_AB = kernel_fn(X=matrix_A, Y=matrix_B)

        return kernel_AA, kernel_BB, kernel_AB, N_A, N_B

    def calculate_distance(
        self,
        data_A: np.ndarray,
        data_B: np.ndarray,
        labels_A: np.ndarray,
        labels_B: np.ndarray,
        embedding_name: str,
        kernel_name: str,
        embedding_kwargs: dict = {},
    ) -> tuple[float, float]:
        """
        Computes maximum mean discrepancy (MMD) for A and B. Given a kernel
        embedding k(X,X'), computes

            MMD^2 = 1/(N_A^2)*sum(k(A,A)) + 1/(N_B^2)*sum(k(B,B)) -
                        2/(N_A * N_B)*sum(k(A,B))

        Note that A and B must be matrices for the MMD. Currently mmd
        will reshape arbitarily size np.ndarrays to matrices. However,
        other options include feature embeddings which will be explored in
        the future.

        Args:
            data_A: The first image dataset
            data_B: The second image dataset
            labels_A: Labels for the first dataset. Unused
            labels_B: Labels for the second dataset. Unused
            embedding_name: What feature embeddings, if any, to use for the
                            input arrays
            kernel_name: An appropriate kernel embedding. See kernel_dict
                        for choices
            embedding_kwargs: Dict of arguments to pass to the embedding function

        Returns:
            float: The maximum mean discrepancy between A and B
        """

        kernel_AA, kernel_BB, kernel_AB, N_A, N_B = self._pre_process_data(
            data_A=data_A,
            data_B=data_B,
            embedding_name=embedding_name,
            kernel_name=kernel_name,
            embedding_kwargs=embedding_kwargs,
        )

        # Compute MMD
        mmd = (
            np.sum(kernel_AA) / (N_A**2)
            + np.sum(kernel_BB) / (N_B**2)
            - 2 * np.sum(kernel_AB) / (N_A * N_B)
        )

        # Return
        return mmd, mmd
