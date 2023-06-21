from math import ceil

import numpy as np
import torch
from sklearn.decomposition import PCA
from torchvision import models, transforms
from umap.umap_ import UMAP

RESIZE = transforms.Resize(299)
CENTER = transforms.CenterCrop(299)
NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


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


def umap(array: np.ndarray, random_seed: int, n_components: int) -> np.ndarray:
    """
    UMAP embedding of array

    Args:
        array: A numpy array of data to be embedded with observations on the first index
        random_seed: An integer to set the random seed of the UMAP algorithm
        n_components: The number of components to return in the embedding

    Returns:
        embed_array: A numpy array of the embedded data
    """
    reducer = UMAP(random_state=random_seed, n_components=n_components)
    reducer.fit(array)
    embed_array = reducer.transform(array)
    return embed_array


def inception(array: np.ndarray) -> np.ndarray:
    """
    Embedding using the Inception V3 neural network with pre-trained weights

    Args:
        array: A numpy array of data to be embedded with observations on the first index

    Returns:
        embed_array: A numpy array of the embedded data
    """

    # Load the inception_v3 pretrained model
    model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
    # Replace the last fully connected layer with an identity layer
    model.fc = torch.nn.Identity()
    # Set model to evaluation mode
    model.eval()

    # Create process to transform data to form required by inception v3
    preprocess = transforms.Compose(
        [
            RESIZE,
            CENTER,
            NORMALIZE,
        ]
    )

    # Process the data to create valid input data for the model
    input_data = preprocess(torch.Tensor(array))

    # move the input and model to GPU for speed, if available
    if torch.cuda.is_available():
        input_data = input_data.to("cuda")
        model.to("cuda")

    # run the data input data through the model in batches
    output_data = []
    for i in range(int(ceil(array.shape[0] / 100))):
        with torch.no_grad():
            output_batch = model(input_data[i * 100 : (i + 1) * 100, :])
            output_data.append(output_batch)
    output_data = torch.cat(output_data, 0)

    # convert the ouptut back to a numpy array
    embed_array = output_data.detach().numpy()

    return embed_array


def pca(array: np.ndarray, n_components: int) -> np.ndarray:
    """
    Sklearn's principal component analysis (PCA) embedding to reduce the
    dimensions of the given array

    Args:
        array: A numpy array of data to be embedded with observations on the first index
        n_components: The number of components to return in the embedding

    Returns:
        embedding: A numpy array of the embedded data

    """
    pca = PCA(svd_solver="full", n_components=n_components)
    pca.fit(array)
    embed_array = pca.transform(array)

    return embed_array


def inception_pca(array: np.ndarray, n_components: int) -> np.ndarray:
    """
    The inception embedding, followed by the PCA embedding

    Args:
        array: A numpy array of data to be embedded with observations on the first index
        n_components: The number of components to return in the embedding

    Returns:
        embedding: A numpy array of the embedded data
    """
    embed_array = inception(array)
    embed_array = pca(embed_array, n_components)
    return embed_array


def inception_umap(
    array: np.ndarray, random_seed: int, n_components: int
) -> np.ndarray:
    """
    The inception embedding, followed by the UMAP embedding

    Args:
        array: A numpy array of data to be embedded with observations on the first index
        random_seed: An integer to set the random seed of the UMAP algorithm
        n_components: The number of components to return in the embedding

    Returns:
        embed_array: A numpy array of the embedded data
    """
    embed_array = inception(array)
    embed_array = umap(embed_array, random_seed, n_components)
    return embed_array


EMBEDDING_FN_DICT = {
    "matrix": array_to_matrix,
    "inception": inception,
    "inception_pca": inception_pca,
    "inception_umap": inception_umap,
}
