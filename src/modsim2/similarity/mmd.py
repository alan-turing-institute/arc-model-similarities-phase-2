import numpy as np
from sklearn import metrics


def array_to_matrix(array: np.ndarray) -> np.ndarray:
    """
    Reshapes arbitrarily sized np.ndarrays to a matrix with the
    same number of observations

    Args:
        array (np.ndarray): An np.ndarray with observations on
                            the first index

    Returns:
        np.ndarray: An np.ndarray with the same number of observations
                    reshaped to a matrix
    """
    shape = array.shape
    matrix = array.reshape(shape[0], np.prod(shape[1:]))
    return matrix


# Dictionary mapping from embedding names to Callables
embedding_dict = {"matrix": array_to_matrix}


# Dictionary mapping from kernel names to Callables
kernel_dict = {
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
        kernel_name: An appropriate kernel embedding. See kernel_dict
                     for choices

    Returns:
        float: The maximum mean discrepancy between A and B
    """
    # Extract embedding callable
    embedding = embedding_dict[embedding_name]

    # In the future, consider optionally using feature embeddings here
    matrix_A = embedding(array_A)
    matrix_B = embedding(array_B)
    N_A = matrix_A.shape[0]
    N_B = matrix_B.shape[0]

    # Extract kernel callable
    kernel = kernel_dict[kernel_name]

    # Compute MMD components
    kernel_AA = kernel(X=matrix_A)
    kernel_BB = kernel(X=matrix_B)
    kernel_AB = kernel(X=matrix_A, Y=matrix_B)

    # Compute MMD
    mmd = (
        np.sum(kernel_AA) / (N_A**2)
        + np.sum(kernel_BB) / (N_B**2)
        - 2 * np.sum(kernel_AB) / (N_A * N_B)
    )

    # Return
    return mmd
