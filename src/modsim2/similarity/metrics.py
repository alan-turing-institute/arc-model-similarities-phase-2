import logging
import platform
from abc import ABC, abstractmethod

import numpy as np
import torch
from otdd.pytorch.distance import DatasetDistance
from sklearn import metrics
from torch.utils.data import TensorDataset

from modsim2.similarity.embeddings import EMBEDDING_FN_DICT


class DistanceMetric(ABC):
    def __init__(self, seed: int = 42):
        torch.manual_seed(seed)

    @abstractmethod
    def _pre_process_data(self):
        pass

    @abstractmethod
    def calculate_metric(
        self,
        data_A: np.ndarray,
        data_B: np.ndarray,
        labels_A: np.ndarray,
        labels_B: np.ndarray,
    ) -> float:
        pass


class MMD(DistanceMetric):
    def __init__(self, seed: int = 42):
        # Kernel dictionary
        self.MMD_KERNEL_DICT = {
            "rbf": metrics.pairwise.rbf_kernel,
            "laplace": metrics.pairwise.laplacian_kernel,
        }

    def _pre_process_data(
        self, data_A: np.ndarray, data_B: np.ndarray, embedding_name: str
    ):
        # Extract embedding callable
        embedding_fn = EMBEDDING_FN_DICT[embedding_name]

        # In the future, consider optionally using feature embeddings here
        matrix_A = embedding_fn(data_A)
        matrix_B = embedding_fn(data_B)
        N_A = matrix_A.shape[0]
        N_B = matrix_B.shape[0]

        return matrix_A, matrix_B, N_A, N_B

    def calculate_metric(
        self,
        data_A: np.ndarray,
        data_B: np.ndarray,
        labels_A: np.ndarray,
        labels_B: np.ndarray,
        embedding_name: str,
        kernel_name: str,
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

        matrix_A, matrix_B, N_A, N_B = self._pre_process_data(
            data_A, data_B, embedding_name
        )

        # Extract kernel callable
        kernel_fn = self.MMD_KERNEL_DICT[kernel_name]

        # Compute MMD components
        logging.info(
            f"Computing {N_A} by {N_A}, {N_B} by {N_B}, and {N_A} by {N_B} kernels. "
            "This may take some time!"
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


class OTDD(DistanceMetric):
    def __init__(self, seed: int = 42):
        super().__init__(seed)

    def _pre_process_data(
        self,
        data_A: np.ndarray,
        data_B: np.ndarray,
        targets_A: np.ndarray,
        targets_B: np.ndarray,
    ):
        dataset_A = TensorDataset(torch.tensor(data_A), torch.tensor(targets_A))
        dataset_B = TensorDataset(torch.tensor(data_B), torch.tensor(targets_B))
        dataset_A.classes = torch.sort(torch.unique(torch.tensor(targets_A))).values
        dataset_B.classes = torch.sort(torch.unique(torch.tensor(targets_B))).values

        return dataset_A, dataset_B

    def calculate_metric(
        self,
        data_A: np.ndarray,
        data_B: np.ndarray,
        labels_A: np.ndarray,
        labels_B: np.ndarray,
        max_samples: int,
        device: str,
        **kwargs,
    ) -> float:
        """
        Calculates the optimal transport distance between datasets

        Args:
            array_A: The first image dataset
            array_B: The second image dataset
            max_samples (int):  maximum number of samples used in outer-level
                        otdd problem.
            device (str): the device on which the calculation will run,
                        e.g. 'cpu', 'mps'

        Returns:
            float: The otdd between A and B
        """
        dataset_A, dataset_B = self._pre_process_data(
            data_A, data_B, labels_A, labels_B
        )
        # the next two lines are a placeholder to ensure that code will run on MacBook
        # with M1 processor - is likely to require changing when issue 45 is addressed
        if platform.processor() == "arm":
            device = "mps"
        kwargs["device"] = device
        dist = DatasetDistance(dataset_A, dataset_B, **kwargs)
        d = dist.distance(maxsamples=max_samples)
        return float(d)
