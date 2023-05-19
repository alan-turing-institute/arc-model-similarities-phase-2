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

    def __init__(self, seed: int):
        super().__init__(seed)

        self.test_proportion = 0.2  # The % of the dataset to be held out for testing
        self.train_balance = (
            "equal"  # Determines ratio of data from A & B in the training data
        )
        self.embedding_name = None  # Name of the embedding function
        self.train_data = None
        self.train_labels = None
        self.test_data = None
        self.test_labels = None
        self.embed_train_data = None
        self.embed_test_data = None
        self.models = []  # List of trained classifiers
        self.errors = []  # List of errors for classifiers on test data

    def __train_test_split(
        self,
        data_A: np.ndarray,
        data_B: np.ndarray,
    ):
        """
        This method splits the two datasets data_A and data_B into train and test
        samples. To calculate the PAD, there should be the same number of samples
        from A & B in the test dataset. However, there may not be the same number
        of samples in A & B. The following lines of code determine the size of the
        test dataset for both A & B - the minimum proportion size of the two datasets

        Args:
            data_A: all records in dataset A (excludes target value)
            data_B: all records in dataset B (excludes target value)

        Returns:
            train_A: records in dataset A to be used for training
            test_A: records in dataset A to be used for testing (final evaluation)
            train_B: records in dataset B to be used for training
            test_B: records in dataset B to be used for testing (final evaluation)
        """

        if self.train_balance == "equal":
            # If the data_A and data_B have different numbers of records then drop
            # records from the larger of the two datasets so that A & B are of
            # equal size
            if data_A.shape[0] > data_B.shape[0]:
                data_A = self._reduce_size_array(data_A, data_B.shape[0])
            elif data_B.shape[0] > data_A.shape[0]:
                data_B = self._reduce_size_array(data_B, data_A.shape[0])
        elif self.train_balance != "ratio":
            raise ValueError(
                "An invalid train_balance value has been provided", self.train_balance
            )

        # Determine the number of samples of test data to be drawn from A&B
        test_size_A = int(np.round(data_A.shape[0] * self.test_proportion))
        test_size_B = int(np.round(data_B.shape[0] * self.test_proportion))

        # Split A&B into train and test
        train_A, test_A = train_test_split(data_A, test_size=test_size_A, shuffle=True)
        train_B, test_B = train_test_split(data_B, test_size=test_size_B, shuffle=True)

        # test_A and test_B must have the same number of records for this metric
        # If one is greater than the other (which may occur when the ratio between A&B
        # has been preserved in the training data) then the excess records will
        # be dropped at random
        if test_A.shape[0] > test_B.shape[0]:
            test_A = self._reduce_size_array(test_A, test_B.shape[0])
        elif test_B.shape[0] > test_A.shape[0]:
            test_B = self._reduce_size_array(test_B, test_A.shape[0])

        return train_A, test_A, train_B, test_B

    @staticmethod
    def _reduce_size_array(array_to_reduce: np.ndarray, new_size: int):
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

    def __concat_data(
        self,
        train_A: np.ndarray,
        test_A: np.ndarray,
        train_B: np.ndarray,
        test_B: np.ndarray,
    ):
        """
        This method will label the A and B train and tests datasets,
        then concatenate the train and test data. The train data are
        then shuffled.
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

    def _pre_process_data(self, data_A: np.ndarray, data_B: np.ndarray):
        """
        Takes A & B datasets as arguments and calls the methods to process
        the data ahead of building the classifiers. First the data are split
        into train and test datasets, then labelled and concatenated and then
        transformed using the embedding function.

        Args:
            data_A: the records in the A dataset (excludes target values)
            data_B: the records in the B dataset (excludes target values)

        Returns:
            embed_train_data: the training data transformed using the embedding function
            embed_test_data: the test data transformed using the embedding function
        """
        # Split A and B into train and test datasets
        # This is done before the data are concatenated in order
        # to ensure that the same proportion of A & B are
        # included in the test dataset
        train_A, test_A, train_B, test_B = self.__train_test_split(data_A, data_B)

        # concatenate A&B datasets
        (
            self.train_data,
            self.train_labels,
            self.test_data,
            self.test_labels,
        ) = self.__concat_data(train_A, test_A, train_B, test_B)

        # Extract embedding callable
        embedding_fn = EMBEDDING_FN_DICT[self.embedding_name]
        # Transform the train and test data using the embedding function
        embed_train_data = embedding_fn(self.train_data)
        embed_test_data = embedding_fn(self.test_data)

        return embed_train_data, embed_test_data

    def __build_models(
        self, kernel_name: str, c_values: list, gamma_values: list, degree_values: list
    ):
        """
        This method will build the classifiers, variation in some hyperparameters
        is allowed as the optimal values are not necessarily known for the datasets.
        The kernel is not varied.
        The trained model are appended to the object's list of models.

        Args:
            kernel_name:
            c_values: list of C values (regularisation param) to be applied
            gamma_values: list of gamma values (kernel coefficient) to be applied
            degree_values: list of degree values for the poly kernel to be applied

        Returns:
            models: list of the models that have been built
        """
        models = []
        for c in c_values:
            for gamma in gamma_values:
                for degree in degree_values:
                    svc = SVC(kernel=kernel_name, C=c, gamma=gamma, degree=degree)
                    svc.fit(X=self.embed_train_data, y=self.train_labels)
                    models.append(svc)
        return models

    def __evaluate_models(self):
        """
        This method will evaluate the classifiers that have been built
        The evaluation metric is the mean absolute error.

        Returns:
            errors: list of errors
        """
        errors = []
        for svc in self.models:
            preds = svc.predict(self.embed_test_data)
            abs_diff = np.abs(preds - self.test_labels)
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
        train_balance="equal",
        gamma_values: list = ["scale"],
        degree_values: list = [3],
    ) -> float:

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
            kernel_name: The kernel to be applied in the SVCs
            embedding_name: What feature embeddings, if any, to use for the
                            input arrays
            test_proportion: the proportion of the dataset to hold out for testing
            train_balance: determines whether the balance between the A&B data
                            in the training data, valid options are 'equal' for
                            a 50:50 split of A&B data, and 'ratio' if the ratio
                            between the original A&B data is to be maintained
            gamma_values: list of gamma values to be applied in SVCs, see sklearn
                            documentation for list of possible values
            degree_values: list of degree values to be applied in polynomial SVCs
        """

        # Set the variables required for the pre-processing
        assert embedding_name in EMBEDDING_FN_DICT, "Error: embedding does not exist"
        self.embedding_name = embedding_name

        # Set the proportion of data to be held out for testing
        assert (
            test_proportion < 1 and test_proportion > 0
        ), "Error: the test_proportion value must be between zero and one"
        self.test_proportion = test_proportion

        # Set whether the proportion of training data from A & B should be equal
        # or maintain the ratio of the the original datasets
        self.train_balance = train_balance

        # Pre-process the data
        self.embed_train_data, self.embed_test_data = self._pre_process_data(
            data_A, data_B
        )

        # Build the classifiers
        self.models = self.__build_models(
            kernel_name, c_values, gamma_values, degree_values
        )

        # Evaluate classifiers
        self.errors = self.__evaluate_models()

        # Compute the proxy a-distance
        min_error = min(self.errors)
        pad = 2 * (1 - 2 * min_error)

        return pad
