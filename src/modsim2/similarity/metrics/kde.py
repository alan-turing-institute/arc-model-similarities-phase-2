import logging

from .metrics import DistanceMetric

# Set module logger
logger = logging.getLogger(__name__)


class KDE(DistanceMetric):
    """
    Class to calculate the distance between two datasets
    using kernel density estimation (KDE).
    """

    def __init__(self, seed: int) -> None:
        pass
