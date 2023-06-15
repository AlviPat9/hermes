"""
Hermes Base Dataset Class
==============

Hermes base class for different dataset classes.

Author: Alvaro Marcos Canedo
"""

from abc import ABC, abstractmethod

import pandas as pd

from hermes.utilities.enums import DatasetType


class BaseDataset(ABC):
    """

    This is the base class for every different dataset. It has been separated in order to create one class per
    different dataset type. Easier to maintain.

    """

    def __init__(self, im):
        """

        Constructor method

        :param im: Dataset type to perform the calculation
        :type im: hermes.utilities.enums.DatasetType
        """
        self.model = self._get_dataset_type(im)

    @abstractmethod
    def prepare_data(self, *args, **kwargs):
        """
        Method to prepare the data for the calculation.

        :param args: Additional arguments for the method.
        :type args: tuple
        :param kwargs: Additional argument for the method.
        :type kwargs: dict

        """
        raise NotImplementedError('As it is an abstract method, it should be overwritten in the appropriate subclass.')

    @staticmethod
    def _get_dataset_type(dataset_type):
        """

        Method to initialize the model type for the different dataset types defined in hermes module.

        :param dataset_type: Dataset type to perform initialization.
        :type dataset_type: hermes.utilities.enums.DatasetType
        :return: Initialization method based on the dataset type input.
        :rtype: various
        """

        if dataset_type == DatasetType.REGRESSION:
            return pd.DataFrame()
        else:
            raise NotImplementedError('Dataset Type not implemented at the moment')


__all__ = ["BaseDataset"]
