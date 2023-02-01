import numpy as np
from sklearn import metrics


def array_to_matrix(array: np.ndarray) -> np.ndarray:
    shape = array.shape
    matrix = array.reshape(shape[0], np.prod(shape[1:]))
    return matrix


def mmd_rbf(array_A: np.ndarray, array_B: np.ndarray) -> float:
    # Transform to matrices
    matrix_A = array_to_matrix(array_A)
    matrix_B = array_to_matrix(array_B)
    N_A = matrix_A.shape[0]
    N_B = matrix_B.shape[0]

    kA = metrics.pairwise.rbf_kernel(X=matrix_A)
    kB = metrics.pairwise.rbf_kernel(X=matrix_B)
    kAB = metrics.pairwise.rbf_kernel(X=matrix_A, Y=matrix_B)

    mmd = (
        np.sum(kA) / (N_A**2)
        + np.sum(kB) / (N_B**2)
        - 2 * np.sum(kAB) / (N_A * N_B)
    )

    return mmd
