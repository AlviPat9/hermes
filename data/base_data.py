"""
Hermes Base Data
===================

This class is the base class for data processing.

Author: Alvaro Marcos Canedo
"""

from hermes.utilities.enums import MissingData, Normalization


class BaseData(object):
    """
    This is the base class for the data treated in Hermes Data Processor.
    :param key: The name of the property, to be used to select the data
    :type key: string
    """

    def __init__(self, key):
        """
        Constructor
        """
        self.key = key

    def set_attribute(self, attr, value):
        """
        Set existing attribute given the name and value.
        :param attr: The attribute name.
        :type attr: string
        :param value: The value to be set.
        :type value: any
        """
        if hasattr(self, attr):
            setattr(self, attr, value)
        else:
            raise AttributeError('{} has no attribute {}'.format(self.key, attr))


class NumericalData(BaseData):
    """
    This is the base class for the Numerical data processing module.


    :param key: Name of the parameter (column) of the dataframe.
    :type key: string
    :param data: dataframe containing data for the analysis.
    :type data: class:`pandas.DataFrame`

    """

    def __init__(self, key, data):
        """

        Constructor method.

        """
        super().__init__(key)

        self.mean_ = data.mean(axis=0)
        self.min_ = data.min(axis=0)
        self.max_ = data.max(axis=0)
        self.std_ = data.std(axis=0)
        self.var_ = self.std_ ** 2
        self.median_ = data.median(axis=0)


class CategoricalData(BaseData):
    """
    This is the base class for the categorical data.

    :param key: The name of the property, to be used to select the data
    :type key: string
    :param data: Dataframe containing the categorical data.
    :type data: class:`pandas.DataFrame`
    """
    def __init__(self, key, data):
        """
        Constructor
        """
        # Call the parent class
        super().__init__(key)

        self.categories = list(data.unique())
        self._encoder()

    def _encoder(self):
        """
        Method to get the ordinal encoder of the categorical column.

        """
        # Assign each variable an integer
        self.encoder = {}
        for i, key in enumerate(self.categories):
            self.encoder[key] = int(i)


__all__ = ['NumericalData', "CategoricalData"]
