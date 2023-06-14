"""
Hermes Enumerators
============

This module includes all the enumerators to be used in Hermes.


"""

from enum import IntEnum, auto


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


class Normalization(IntEnum):
    """
    This class defines how to normalize/Standarize values from the database.
    It is important for MachineLearning Techniques and regressions.

    """
    MINMAX = auto()
    STD = auto()
    L1 = auto()
    L2 = auto()


__all__ = ["Normalization", "MissingData"]
