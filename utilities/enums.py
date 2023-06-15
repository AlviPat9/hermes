"""
Hermes Enumerators
============

This module includes all the enumerators to be used in Hermes.


"""

from enum import IntEnum, auto


class DatasetType(IntEnum):
    """

    This class sets all available dataset types.

    """
    # Numerical and categorical data for regression, classification models, decision trees... are grouped by regression
    REGRESSION = auto()
    IMAGE_PROCESSOR = auto()


class MissingData(IntEnum):
    """
    This class defines how to fill the missing values from the database
    Default -> MEAN

    """
    MEAN = auto()
    DELETE = auto()
    #   STD MIN & MAX should not be used
    STD = auto()
    MIN = auto()
    MAX = auto()
    MODE = auto()


class Normalization(IntEnum):
    """
    This class defines how to normalize/Standarize values from the database.
    It is important for MachineLearning Techniques and regressions.

    """
    MINMAX = auto()
    STD = auto()
    L1 = auto()
    L2 = auto()
    ROBUST_SCALER = auto()


class CategoricalData(IntEnum):
    """

    This class defines the different types of categirical data formatting that can be applied to the database.

    """
    DUMMY = auto()
    ONE_HOT = auto()


__all__ = ["DatasetType", "Normalization", "MissingData", "CategoricalData"]
