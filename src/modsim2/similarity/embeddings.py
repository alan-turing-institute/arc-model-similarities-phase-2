import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from torchvision import models, transforms
from umap.umap_ import UMAP

from modsim2.utils.accelerator import choose_auto_accelerator

# Constants
RESIZE = transforms.Resize(299)
CENTER = transforms.CenterCrop(299)
NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
INCEPTION_NFEATS = 2048


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


def inception(
    array: np.ndarray,
    batch_size: int,
    device: str,
) -> np.ndarray:
    """
    Embedding using the Inception V3 neural network with pre-trained weights

    Args:
        array: A numpy array of data to be embedded with observations on the first index
        batch_size: Batch size for the dataloader
        device: Which device to use. If set to "auto" will attempt to use GPU.

    Returns:
        embed_array: A numpy array of the embedded data
    """

    # Set device if auto
    if device == "auto":
        device = choose_auto_accelerator()

    # Load the inception_v3 pretrained model
    model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
    # Replace the last fully connected layer with an identity layer
    model.fc = torch.nn.Identity()
    # Set model to evaluation mode
    model.eval()

    # Move model to device - data moved to device in batches later
    model.to(device)

    # Get the dimensions
    n_samples = array.shape[0]

    # Create process to transform data to form required by inception v3
    preprocess = transforms.Compose(
        [
            RESIZE,
            CENTER,
            NORMALIZE,
        ]
    )

    # create output
    output_data = torch.zeros((n_samples, INCEPTION_NFEATS))

    # run the data input data through the model in batches
    data_loader = DataLoader(
        array,
        batch_size=batch_size,
        shuffle=False,
    )
    with torch.inference_mode():
        for i, batch in enumerate(data_loader):
            batch = preprocess(batch)
            # Move data and model to device
            batch = batch.to(device)
            output_batch = model(batch)
            slice = i * batch_size
            output_data[
                slice : (slice + output_batch.shape[0])
            ] = output_batch.cpu().detach()

    # Return
    return output_data.numpy()


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


def inception_pca(
    array: np.ndarray,
    batch_size: int,
    device: str,
    n_components: int,
) -> np.ndarray:
    """
    The inception embedding, followed by the PCA embedding

    Args:
        array: A numpy array of data to be embedded with observations on the first index
        batch_size: Batch size for the dataloader
        device: Which device to use. If set to "auto" will attempt to use GPU.
        n_components: The number of components to return in the embedding

    Returns:
        embedding: A numpy array of the embedded data
    """
    embed_array = inception(
        array=array,
        batch_size=batch_size,
        device=device,
    )
    embed_array = pca(
        array=embed_array,
        n_components=n_components,
    )
    return embed_array


def inception_umap(
    array: np.ndarray,
    batch_size: int,
    device: str,
    random_seed: int,
    n_components: int,
) -> np.ndarray:
    """
    The inception embedding, followed by the UMAP embedding

    Args:
        array: A numpy array of data to be embedded with observations on the first index
        batch_size: Batch size for the dataloader
        device: Which device to use. If set to "auto" will attempt to use GPU.
        random_seed: An integer to set the random seed of the UMAP algorithm
        n_components: The number of components to return in the embedding

    Returns:
        embed_array: A numpy array of the embedded data
    """
    embed_array = inception(
        array=array,
        batch_size=batch_size,
        device=device,
    )
    embed_array = umap(
        array=embed_array,
        random_seed=random_seed,
        n_components=n_components,
    )
    return embed_array


EMBEDDING_FN_DICT = {
    "matrix": array_to_matrix,
    "inception": inception,
    "inception_pca": inception_pca,
    "inception_umap": inception_umap,
}
