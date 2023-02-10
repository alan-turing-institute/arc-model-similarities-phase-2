import numpy as np

from modsim2.similarity.embeddings import array_to_matrix


def test_array_to_matrix():
    dim = (10, 10, 10, 10)
    arr = np.random.rand(*dim)
    mat = array_to_matrix(arr)
    assert mat.shape == (arr.shape[0], np.prod(arr.shape[1:]))  # tests correct dims
    assert all(mat[:, 0] == arr[:, 0, 0, 0])  # tests preservation of first dim
