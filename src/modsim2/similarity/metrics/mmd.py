import logging

import numpy as np
from sklearn import metrics

from modsim2.similarity.embeddings import EMBEDDING_FN_DICT

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
    ):
        # Extract embedding callable
        embedding_fn = EMBEDDING_FN_DICT[embedding_name]

        # In the future, consider optionally using feature embeddings here
        matrix_A = embedding_fn(data_A)
        matrix_B = embedding_fn(data_B)
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
    ) -> float:
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
            embedding_name: What feature embeddings, if any, to use for the
                            input arrays
            kernel_name: An appropriate kernel embedding. See kernel_dict
                        for choices

        Returns:
            float: The maximum mean discrepancy between A and B
        """

        kernel_AA, kernel_BB, kernel_AB, N_A, N_B = self._pre_process_data(
            data_A, data_B, embedding_name, kernel_name
        )

        # Compute MMD
        mmd = (
            np.sum(kernel_AA) / (N_A**2)
            + np.sum(kernel_BB) / (N_B**2)
            - 2 * np.sum(kernel_AB) / (N_A * N_B)
        )

        # Return
        return mmd
