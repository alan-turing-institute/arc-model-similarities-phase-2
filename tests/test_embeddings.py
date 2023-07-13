import numpy as np

import modsim2.similarity.embeddings as embeddings


def test_array_to_matrix():
    dim = (10, 10, 10, 10)
    arr = np.random.rand(*dim)
    mat = embeddings.array_to_matrix(arr)
    assert mat.shape == (arr.shape[0], np.prod(arr.shape[1:]))  # tests correct dims
    assert all(mat[:, 0] == arr[:, 0, 0, 0])  # tests preservation of first dim


def test_umap():
    dim = (10, 10)
    arr = np.random.rand(*dim)
    num_components = 2
    embed = embeddings.umap(arr, 42, num_components)
    assert embed.shape == (arr.shape[0], num_components)  # tests correct dims


def test_pca():
    dim = (10, 10)
    arr = np.random.rand(*dim)
    num_components = 2
    embed = embeddings.pca(arr, 2)
    assert embed.shape == (arr.shape[0], num_components)  # tests correct dims


def test_inception():
    dim = (10, 3, 10, 10)
    arr = np.random.rand(*dim).astype(np.float32)
    embed = embeddings.inception(arr, batch_size=5, device="cpu")
    assert embed.shape == (arr.shape[0], 2048)  # tests correct dims


def test_inception_pca():
    dim = (10, 3, 10, 10)
    arr = np.random.rand(*dim).astype(np.float32)
    num_components = 2
    embed = embeddings.inception_pca(
        arr, batch_size=5, device="cpu", n_components=num_components
    )
    assert embed.shape == (arr.shape[0], num_components)  # tests correct dims


def test_inception_umap():
    dim = (10, 3, 10, 10)
    arr = np.random.rand(*dim).astype(np.float32)
    num_components = 2
    embed = embeddings.inception_umap(
        arr, batch_size=5, device="cpu", random_seed=42, n_components=num_components
    )
    assert embed.shape == (arr.shape[0], num_components)  # tests correct dims
