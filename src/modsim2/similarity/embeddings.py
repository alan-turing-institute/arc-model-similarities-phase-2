import numpy as np


def array_to_matrix(array: np.ndarray) -> np.ndarray:
    """
    Reshapes np.ndarrays of >=2 dimensions to a 2-dimensional
    np.ndarray, while preserving the first dimension

    Args:
        array (np.ndarray): An np.ndarray with observations on
                            the first index

    Returns:
        np.ndarray: An np.ndarray with the same number of observations
                    reshaped to a 2 dimension
    """
    shape = array.shape
    matrix = array.reshape((shape[0], -1))
    return matrix


EMBEDDING_FN_DICT = {"matrix": array_to_matrix}
