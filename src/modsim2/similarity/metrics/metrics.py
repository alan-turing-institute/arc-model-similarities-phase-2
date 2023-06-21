from abc import ABC, abstractmethod

import numpy as np
from pytorch_lightning.utilities.seed import seed_everything

from modsim2.similarity.embeddings import EMBEDDING_FN_DICT


class DistanceMetric(ABC):
    def __init__(self, metric_seed: int):
        seed_everything(metric_seed)

    @abstractmethod
    def _pre_process_data(self, data_A: np.ndarray, data_B: np.ndarray):
        """
        This method will take the source and target data
        and will process it into a format so that the distance
        can be calculated. The method will return the processed
        data.

        The steps for processing the data will vary by subclass
        as will the format in which the data are returned.
        """
        pass

    def _embed_data(
        self,
        data_A: np.ndarray,
        data_B: np.ndarray,
        embedding_name: str,
        embedding_kwargs: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        This method will embed the source and target data using a function.
        For all embeddings, the data are returned as 2-dimensional matrices
        with shape (num_records, num_features)

        Args:
            data_A: numpy array of data
            data_B: numpy array of data
            embedding_name: the name of the embedding function
            embedding_args: a dictionary of arguments to pass to the embedding function
                            (may be empty)

        Returns:
            embed_A: the embedded representation of data_A
            embed_B: the embedded representation of data_B
        """

        # Extract embedding callable
        embedding_fn = EMBEDDING_FN_DICT[embedding_name]

        # Apply embeddigns
        embed_A = embedding_fn(data_A, **embedding_kwargs)
        embed_B = embedding_fn(data_B, **embedding_kwargs)

        return embed_A, embed_B

    @abstractmethod
    def calculate_distance(
        self,
        data_A: np.ndarray,
        data_B: np.ndarray,
        labels_A: np.ndarray,
        labels_B: np.ndarray,
    ) -> float:
        """
        This method takes the source and target data and labels
        as numpy arrays and returns the distance metric.
        """
        pass
