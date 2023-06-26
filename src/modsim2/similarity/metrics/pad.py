import logging

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from modsim2.similarity.embeddings import EMBEDDING_FN_DICT

from .metrics import DistanceMetric

# Set module logger
logger = logging.getLogger(__name__)


class PAD(DistanceMetric):
    """
    Class to calculate the proxy a-distance between two datasets.
    The metric combines the source and target (A and B) datasets.
    The combined dataset is divided into train and test sets,
    and classifier(s) are built using the train dataset to distinguish
    between the source and target. The classifier(s) are then evaluated
    using the test dataset and the lowest error across all the classifiers
    is used to calculate the proxy a-distance:
    pad = 2 * (1 - 2 * min_error)

    In this implementation, Support Vector Machines are used as
    the classifiers. The classifiers can be created with any of the
    kernels allowed in sklearn's SVC, and the data can be transformed
    with any of the embeddings available in this project. Multiple
    classifiers may be created to allow for some hyperparameter tuning.

    """

    def __init__(self, seed: int) -> None:
        super().__init__(seed)

    def _train_test_split(
        self,
        data_A: np.ndarray,
        data_B: np.ndarray,
        test_proportion: float,
        balance_train: bool,
        balance_test: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        This method splits the two datasets data_A and data_B into train and test
        samples. To calculate the PAD, there should be the same number of samples
        from A & B in the test dataset. However, there may not be the same number
        of samples in A & B. The following lines of code determine the size of the
        test dataset for both A & B - the minimum proportion size of the two datasets

        Args:
            data_A: all records in dataset A (excludes target value)
            data_B: all records in dataset B (excludes target value)
            test_proportion: The proportion of the dataset to hold out for testing.
                             Note that if balance_test is true, the number of examples
                             held from both A and B's test sets will be the same, and
                             will correspond to the smallest dataset.
            balance_train: Determines whether to balance the number of observations
                           from A and B in the training dataset
            balance_test: Determines whether to balance the number of observations
                          from A and B in the test dataset

        Returns:
            train_A: records in dataset A to be used for training
            test_A: records in dataset A to be used for testing (final evaluation)
            train_B: records in dataset B to be used for training
            test_B: records in dataset B to be used for testing (final evaluation)
        """

        # Determine the number of samples of test data to be drawn from A&B
        test_size_A = int(np.round(data_A.shape[0] * test_proportion))
        test_size_B = int(np.round(data_B.shape[0] * test_proportion))

        # Split A&B into train and test
        train_A, test_A = train_test_split(data_A, test_size=test_size_A, shuffle=True)
        train_B, test_B = train_test_split(data_B, test_size=test_size_B, shuffle=True)

        if balance_train:
            # User specified option to balance the training dataset. Prevents imbalance
            # in the training dataset from driving the behaviour of the classifier
            train_A, train_B = self._balance_datasets(train_A, train_B)

        if balance_test:
            # User-specifed option to balance test set, to prevent accurately predicting
            # the larger dataset from skewing the metric
            test_A, test_B = self._balance_datasets(test_A, test_B)

        return train_A, test_A, train_B, test_B

    def _balance_datasets(
        self,
        data_A: np.ndarray,
        data_B: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        A function that takes two datasets as input, filters the larger dataset
        down to the size of the smaller dataset, and then returns them.

        Args:
            data_A: first dataset
            data_B: second dataset

        Returns: Both datasets, one of which may have been resized
        """
        if data_A.shape[0] > data_B.shape[0]:
            data_A = self._sample_from_array(data_A, data_B.shape[0])
        elif data_B.shape[0] > data_A.shape[0]:
            data_B = self._sample_from_array(data_B, data_A.shape[0])
        return data_A, data_B

    @staticmethod
    def _sample_from_array(array_to_reduce: np.ndarray, new_size: int) -> np.ndarray:
        """
        Resizes a numpy array by dropping random rows of data based
        on the first dimension of the array

        Args:
            array_to_reduce: numpy array that is to be resized
            new_size: the number of rows of data that the array will
                      be reduced to

        Returns:
            reduced_array:
        """
        random_indices = np.random.choice(
            array_to_reduce.shape[0], new_size, replace=False
        )
        reduced_array = array_to_reduce[random_indices, :]

        return reduced_array

    def _concat_data(
        self,
        train_A: np.ndarray,
        test_A: np.ndarray,
        train_B: np.ndarray,
        test_B: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        This method will label the A and B train and tests datasets,
        then concatenate the train and test data.
        The data from A will be labelled 1 and B will be labelled 0
        The train and test data and labels are assigned to class attributes

        Args:
            train_A: the data for training from dataset A
            test_A: the data for testing from dataset A
            train_B: the data for training from dataset B
            test_B: the data for testing from dataset B

        Returns:
            train_data: the concatenated training data
            train_labels: the concatenated labels for the training data
            test_data: the concatenated test data
            test_labels: the concatenated labels for the test data
        """
        # Create labels for A & B data
        train_labels_A = np.ones(train_A.shape[0])
        test_labels_A = np.ones(test_A.shape[0])
        train_labels_B = np.zeros(train_B.shape[0])
        test_labels_B = np.zeros(test_B.shape[0])

        # Concatenate the A and B datasets into train and test
        train_data = np.concatenate((train_A, train_B), axis=0)
        train_labels = np.concatenate((train_labels_A, train_labels_B), axis=0)
        test_data = np.concatenate((test_A, test_B), axis=0)
        test_labels = np.concatenate((test_labels_A, test_labels_B), axis=0)

        return train_data, train_labels, test_data, test_labels

    def _pre_process_data(
        self,
        data_A: np.ndarray,
        data_B: np.ndarray,
        test_proportion: float,
        balance_train: bool,
        balance_test: bool,
        embedding_name: str,
        embedding_kwargs: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Takes A & B datasets as arguments and calls the methods to process
        the data ahead of building the classifiers. First the data are split
        into train and test datasets, then labelled and concatenated and then
        transformed using the embedding function.

        Args:
            data_A: the records in the A dataset (excludes target values)
            data_B: the records in the B dataset (excludes target values)
            test_proportion: The proportion of the dataset to hold out for testing.
                             Note that if balance_test is true, the number of examples
                             held from both A and B's test sets will be the same, and
                             will correspond to the smallest dataset.
            balance_train: Determines whether to balance the number of observations
                           from A and B in the training dataset
            balance_test: Determines whether to balance the number of observations
                          from A and B in the test dataset
            embedding_name: What feature embeddings, if any, to use for the
                            input arrays

        Returns:
            embed_train_data: the training data transformed using the embedding function
            train_labels: the labels for the training data
            embed_test_data: the test data transformed using the embedding function
            test_labels: the labels for the test data
        """
        # Split A and B into train and test datasets
        # This is done before the data are concatenated in order
        # to ensure that the same proportion of A & B are
        # included in the test dataset
        train_A, test_A, train_B, test_B = self._train_test_split(
            data_A=data_A,
            data_B=data_B,
            test_proportion=test_proportion,
            balance_train=balance_train,
            balance_test=balance_test,
        )

        # concatenate A&B datasets
        train_data, train_labels, test_data, test_labels = self._concat_data(
            train_A=train_A, test_A=test_A, train_B=train_B, test_B=test_B
        )

        embed_train_data, embed_test_data = self._embed_data(
            data_A=train_data,
            data_B=test_data,
            embedding_name=embedding_name,
            embedding_kwargs=embedding_kwargs,
        )

        return embed_train_data, train_labels, embed_test_data, test_labels

    def _build_models(
        self,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        kernel_name: str,
        c_values: list,
        gamma_values: list,
        degree_values: list,
    ) -> list[SVC]:
        """
        This method will build the classifiers, variation in some hyperparameters
        is allowed as the optimal values are not necessarily known for the datasets.
        The kernel is not varied.
        The trained model are appended to the object's list of models.

        Args:
            train_data: Dataset to use in training the classifiers
            train_labels: Labels to use in training the classifiers
            kernel_name: Kernel to use. Can be 'linear', 'poly', or 'rbf'
            c_values: List of C values (regularisation param) to be applied
            gamma_values: List of gamma values (kernel coefficient) to be applied
            degree_values: List of degree values for the poly kernel to be applied

        Returns:
            models: list of the models that have been built
        """
        models = []
        for c in c_values:
            for gamma in gamma_values:
                for degree in degree_values:
                    svc = SVC(kernel=kernel_name, C=c, gamma=gamma, degree=degree)
                    svc.fit(X=train_data, y=train_labels)
                    models.append(svc)
        return models

    def _evaluate_models(
        self,
        models: list[SVC],
        test_data: np.ndarray,
        test_labels: np.ndarray,
    ) -> list[float]:
        """
        This method will evaluate the classifiers that have been built
        The evaluation metric is the mean absolute error.

        Returns:
            errors: list of errors
        """
        errors = []
        for svc in models:
            preds = svc.predict(test_data)
            abs_diff = np.abs(preds - test_labels)
            error = np.sum(abs_diff) / len(preds)
            errors.append(error)
        return errors

    def calculate_distance(
        self,
        data_A: np.ndarray,
        data_B: np.ndarray,
        labels_A: np.ndarray,
        labels_B: np.ndarray,
        c_values: list,
        kernel_name: str,
        embedding_name: str,
        test_proportion: float = 0.2,
        balance_train: bool = True,
        balance_test: bool = True,
        gamma_values: list = ["scale"],
        degree_values: list = [3],
        embedding_args: dict = {},
    ) -> tuple[float, float]:

        """
        Calculates the proxy a-distance for datasets A and B.
        As this implementation uses SVM, the attributes include
        some parameters in sklearn's implementation of SVC. A
        single value for the kernel name is accepted, and lists
        of possible C values, gamma values and degree values
        (to allow for some hyperparameter tuning).

        Defaults are provided for the gamma and degree values
        meaning these do not have to be supplied in the metrics
        config.

        Args:
            data_A: The first image dataset
            data_B: The second image dataset
            labels_A: The target values for the first dataset
            labels_B: The target values for the second dataset
            c_values: List of C values (regularisation param) to be applied
                      in the SVCs
            kernel_name: The kernel to be applied in the SVCs. Can be 'linear', 'poly',
                         or 'rbf'
            embedding_name: What feature embeddings, if any, to use for the
                            input arrays
            test_proportion: The proportion of the dataset to hold out for testing.
                             Note that if balance_test is true, the number of examples
                             held from both A and B's test sets will be the same, and
                             will correspond to the smallest dataset.
            balance_train: Determines whether to balance the number of observations
                           from A and B in the training dataset
            balance_test: Determines whether to balance the number of observations
                          from A and B in the test dataset
            gamma_values: List of gamma values to be applied in SVCs, see sklearn
                            documentation for list of possible values
            degree_values: List of degree values to be applied in polynomial SVCs
            embedding_args: Dict of arguments to pass to the embedding function
        """

        # Check for valid embedding choice
        assert embedding_name in EMBEDDING_FN_DICT, "Error: embedding does not exist"

        # Check for valid test proportion choice
        assert (
            test_proportion < 1 and test_proportion > 0
        ), "Error: the test_proportion value must be between zero and one"

        # Pre-process the data
        train_data, train_labels, test_data, test_labels = self._pre_process_data(
            data_A=data_A,
            data_B=data_B,
            test_proportion=test_proportion,
            balance_train=balance_train,
            balance_test=balance_test,
            embedding_name=embedding_name,
            embedding_args=embedding_args,
        )

        # Build the classifiers
        models = self._build_models(
            train_data=train_data,
            train_labels=train_labels,
            kernel_name=kernel_name,
            c_values=c_values,
            gamma_values=gamma_values,
            degree_values=degree_values,
        )

        # Evaluate classifiers
        errors = self._evaluate_models(
            models=models,
            test_data=test_data,
            test_labels=test_labels,
        )

        # Compute the proxy a-distance
        min_error = min(errors)
        pad = 2 * (1 - 2 * min_error)

        return pad, pad
