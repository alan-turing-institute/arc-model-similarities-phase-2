import logging

import numpy as np
from sklearn import metrics

from modsim2.similarity.embeddings import EMBEDDING_FN_DICT

# Set module logger
logger = logging.getLogger(__name__)

# Kernel dictionary
MMD_KERNEL_DICT = {
    "rbf": metrics.pairwise.rbf_kernel,
    "laplace": metrics.pairwise.laplacian_kernel,
}


def mmd(
    array_A: np.ndarray, array_B: np.ndarray, embedding_name: str, kernel_name: str
) -> float:
    """
    Computes maximum mean discrepancy (MMD) for A and B. Given a kernel
    embedding k(X,X'), computes

        MMD^2 = 1/(N_A^2)*sum(k(A,A)) + 1/(N_B^2)*sum(k(B,B)) -
                    2/(N_A * N_B)*sum(k(A,B))

    Note that A and B must be matrices for the MMD. Currently mmd()
    will reshape arbitarily size np.ndarrays to matrices. However,
    other options include feature embeddings which will be explored in
    the future.

    Args:
        array_A: The first image dataset
        array_B: The second image dataset
        embedding_name: What feature embeddings, if any, to use for the
                        input arrays
        kernel_name: An appropriate kernel embedding. See kernel_dict
                     for choices

    Returns:
        float: The maximum mean discrepancy between A and B
    """
    # Extract embedding callable
    embedding_fn = EMBEDDING_FN_DICT[embedding_name]

    # In the future, consider optionally using feature embeddings here
    matrix_A = embedding_fn(array_A)
    matrix_B = embedding_fn(array_B)
    N_A = matrix_A.shape[0]
    N_B = matrix_B.shape[0]

    # Extract kernel callable
    kernel_fn = MMD_KERNEL_DICT[kernel_name]

    # Compute MMD components
    logging.info(
        "Computing "
        + str(N_A)
        + " by "
        + str(N_A)
        + ", "
        + str(N_B)
        + " by "
        + str(N_B)
        + ", and "
        + str(N_A)
        + " by "
        + str(N_B)
        + " kernels. This may take some time!"
    )
    kernel_AA = kernel_fn(X=matrix_A)
    kernel_BB = kernel_fn(X=matrix_B)
    kernel_AB = kernel_fn(X=matrix_A, Y=matrix_B)

    # Compute MMD
    mmd = (
        np.sum(kernel_AA) / (N_A**2)
        + np.sum(kernel_BB) / (N_B**2)
        - 2 * np.sum(kernel_AB) / (N_A * N_B)
    )

    # Return
    return mmd
