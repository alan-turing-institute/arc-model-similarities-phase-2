import logging

import numpy as np
from sklearn.model_selection import train_test_split

# from sklearn import metrics
from sklearn.svm import SVC
from sklearn.utils import shuffle

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

        self.__test_proportion = 0.2  # The % of the dataset to be held out for testing
        self.__embedding_name = None  # Name of the embedding function
        self.__train_data = None
        self.__train_labels = None
        self.__test_data = None
        self.__test_labels = None
        self.__embed_train_data = None
        self.__embed_test_data = None
        self.__models = []  # List of trained classifiers
        self.__errors = []  # List of errors for classifiers on test data

    def get_test_labels(self) -> np.ndarray:
        """
        This method is required for one of the tests - not used in the calculation
        of the metric

        Returns
            self.__test_labels - the labels for the test datasest (a combination of
            both A & B data)
        """
        return self.__test_labels

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
        # Determine the number of samples of test data to be drawn from A&B
        test_size_A = np.round(data_A.shape[0] * self.__test_proportion)
        test_size_B = np.round(data_B.shape[0] * self.__test_proportion)
        test_size = int(min(test_size_A, test_size_B))
        # Split A&B into train and test
        train_A, test_A = train_test_split(data_A, test_size=test_size, shuffle=True)
        train_B, test_B = train_test_split(data_B, test_size=test_size, shuffle=True)

        return train_A, test_A, train_B, test_B

    @staticmethod
    def __label_data(data: np.ndarray, label_value: int) -> np.ndarray:
        """
        Takes an array of data and an integer label value. Returns
        an array of the label value with length matching the first
        dimension of the array of data

        Args:
            data: array of data (any dimension)
            label_value: integer value to set the label data

        Returns:
            labels: array of value label_value with length matching
                    the number of data records
        """
        labels = np.ones(data.shape[0])
        labels *= label_value
        return labels

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
        """
        # Create labels for A & B data
        train_labels_A, test_labels_A = self.__label_data(
            train_A, 1
        ), self.__label_data(test_A, 1)
        train_labels_B, test_labels_B = self.__label_data(
            train_B, 0
        ), self.__label_data(test_B, 0)

        # Concatenate the A and B datasets into train and test
        self.__train_data = np.concatenate((train_A, train_B), axis=0)
        self.__train_labels = np.concatenate((train_labels_A, train_labels_B), axis=0)
        self.__test_data = np.concatenate((test_A, test_B), axis=0)
        self.__test_labels = np.concatenate((test_labels_A, test_labels_B), axis=0)
        # Shuffle the train data
        self.__train_data, self.__train_labels = shuffle(
            self.__train_data, self.__train_labels
        )

    def __transform_data(self):
        """
        Method to transform the object's train and test data
        using the object's embedding function
        The transformed data are assigned to attributes of the object
        """
        # Extract embedding callable
        embedding_fn = EMBEDDING_FN_DICT[self.__embedding_name]
        # Transform the train and test data using the embedding function
        self.__embed_train_data = embedding_fn(self.__train_data)
        self.__embed_test_data = embedding_fn(self.__test_data)

    def _pre_process_data(self, data_A: np.ndarray, data_B: np.ndarray):
        """
        Takes A & B datasets as arguments and calls the methods to process
        the data ahead of building the classifiers. First the data are split
        into train and test datasets, then labelled and concatenated and then
        transformed using the embedding function.

        Args:
            data_A: the records in the A dataset (excludes target values)
            data_B: the records in the B dataset (excludes target values)
        """
        # Split A and B into train and test datasets
        # This is done before the data are concatenated in order
        # to ensure that the same proportion of A & B are
        # included in the test dataset
        train_A, test_A, train_B, test_B = self.__train_test_split(data_A, data_B)
        # transform data - will depend on classifier type as to the transform

        self.__concat_data(train_A, test_A, train_B, test_B)

        self.__transform_data()

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
        """
        for c in c_values:
            for gamma in gamma_values:
                for degree in degree_values:
                    svc = SVC(kernel=kernel_name, C=c, gamma=gamma, degree=degree)
                    svc.fit(X=self.__embed_train_data, y=self.__train_labels)
                    self.__models.append(svc)

    def __evaluate_models(self):
        """
        This method will evaluate the classifiers that have been built
        The evaluation metric is the mean absolute error.
        """
        for svc in self.__models:
            preds = svc.predict(self.__embed_test_data)
            errors = np.abs(preds - self.__test_labels)
            error = np.sum(errors) / len(preds)
            self.__errors.append(error)

    def calculate_distance(
        self,
        data_A: np.ndarray,
        data_B: np.ndarray,
        labels_A: np.ndarray,
        labels_B: np.ndarray,
        c_values: list,
        kernel_name: str,
        embedding_name: str,
        test_proportion: float,
        gamma_values: list = ["scale"],
        degree_values: list = [3],
        distinct_classes: bool = False,
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
            gamma_values: list of gamma values to be applied in SVCs
            degree_values: list of degree values to be applied in polynomial SVCs
            distinct_classes: only used during testing - drops records in A & B to
                              so that they have distinct classes
        """

        # Set the variables required for the pre-processing
        assert embedding_name in EMBEDDING_FN_DICT, "Error: embedding does not exist"
        self.__embedding_name = embedding_name

        assert (
            test_proportion < 1 and test_proportion > 0
        ), "The test_proportion value must be between zero and one"
        self.__test_proportion = test_proportion

        if distinct_classes:
            data_A, data_B = self._divide_by_class(data_A, labels_A, data_B, labels_B)

        # Pre-process the data
        self._pre_process_data(data_A, data_B)

        # Build the classifiers
        self.__build_models(kernel_name, c_values, gamma_values, degree_values)

        # Evaluate classifiers
        self.__evaluate_models()

        # Compute the proxy a-distance
        min_error = min(self.__evaluations)
        pad = 2 * (1 - 2 * min_error)

        return pad

    @staticmethod
    def _divide_by_class(
        data_A: np.ndarray,
        labels_A: np.ndarray,
        data_B: np.ndarray,
        labels_B: np.ndarray,
    ):
        """
        This method is only used during testing to create two datasets
        with distinct classes in their target values.
        Takes data records A & B and corresponding arraysof labels.
        It identifies the unique values (classes) across both label arrays.
        Half the classes are randomly selected to be applied in dataset A
        and half in dataset B, the data records are then filtered accordingly.

        Args:
            data_A: The first image dataset
            data_B: The second image dataset
            labels_A: The target values for the first dataset
            labels_B: The target values for the second dataset

        Returns:
            subset_data_A: The first dataset filtered by half of the target values
            subset_data_B: The second dataset filtered by (the other) half of the target
                           values
        """
        all_labels = np.concatenate([np.array(labels_A), np.array(labels_B)])
        unique_classes = np.unique(all_labels)
        np.random.shuffle(unique_classes)
        num_classes_A = int(round(len(unique_classes) / 2, 0))
        classes_A = np.isin(labels_A, unique_classes[:num_classes_A])
        classes_B = np.isin(labels_B, unique_classes[num_classes_A:])
        subset_data_A, subset_data_B = data_A[classes_A], data_B[classes_B]
        return subset_data_A, subset_data_B
