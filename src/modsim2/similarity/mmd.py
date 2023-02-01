from typing import Callable

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


def mmd(array_A: np.ndarray, array_B: np.ndarray, kernel: Callable) -> float:
    """
    Computes maximum mean discrepancy (MMD) for A and B. Given a kernel
    embedding k(X,X'), computes

        MMD^2 = 1/(N_A^2)*sum(k(A,A) + 1/(N_B^2)*sum(k(B,B)) -
                    2/(N_A * N_B)*sum(k(A,B)

    Note that A and B must be matrices for the MMD. Currently mmd()
    will reshape arbitarily size np.ndarrays to matrices. However,
    other options include feature embeddings which will be explored in
    the future.

    Args:
        array_A (np.ndarray): The first image dataset
        array_B (np.ndarray): The second image dataset
        kernel (Callable): An appropriate kernel embedding. See
                           sklearn.metrics.pairwise for choices

    Returns:
        float: The maximum mean discrepancy between A and B
    """
    # In the future, consider using feature embeddings here
    matrix_A = array_to_matrix(array_A)
    matrix_B = array_to_matrix(array_B)
    N_A = matrix_A.shape[0]
    N_B = matrix_B.shape[0]

    kernel_AA = kernel(X=matrix_A)
    kernel_BB = kernel(X=matrix_B)
    kernel_AB = kernel(X=matrix_A, Y=matrix_B)

    mmd = (
        np.sum(kernel_AA) / (N_A**2)
        + np.sum(kernel_BB) / (N_B**2)
        - 2 * np.sum(kernel_AB) / (N_A * N_B)
    )

    return mmd


def mmd_rbf(array_A: np.ndarray, array_B: np.ndarray) -> float:
    """
    A wrapper function for mmd(). Uses the gaussian (radial basis
    function) kernel.

    Args:
        array_A (np.ndarray): A num_obs_A by num_features_A matrix
        array_B (np.ndarray): A num_obs_B by num_features_B matrix

    Returns:
        float: The maximum mean discrepancy between A and B
    """
    return mmd(array_A=array_A, array_B=array_B, kernel=metrics.pairwise.rbf_kernel)
