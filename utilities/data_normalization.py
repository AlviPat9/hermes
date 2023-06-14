"""

Hermes Data Normalization functions module
=================================

This file contains all utilities related to the process of data preparation.

"""


def minmax(data, min_, max_, reverse=False):
    """
    Function for data normalization. Consists of the MinMax formula.
    (value - min) / (max -  min)

    :param data: data process.
    :type data: class: `pandas.core.series.Series`
    :param min_: Minimum value of the dataset.
    :type min_: class:`pandas.core.series.Series`
    :param max_: Maximum value of the dataset.
    :type max_: class:`pandas.core.series.Series`
    :param reverse: Boolean parameter to determine if tuning of reverse tuning is dode.
    :type reverse: bool, optional

    :return dataset transformed
    :rtype class:`pandas.core.series.Series`

    """

    if reverse:
        return data * (max_ - min_) + min_
    else:
        return (data - min_) / (max_ - min_)


def L1(data, raw_data=None, reverse=False):
    """
    Function for data normalization. Consists of the L1 normalization formula.
    value / sum(abs(value))

    :param data: data to process.
    :type data: class: `pandas.core.series.Series`
    :param raw_data: Raw data of the dataset. Only required for reverse transformation.
    :type mean_: class:`pandas.core.series.Series`, optional
    :param reverse: Boolean parameter to determine if tuning of reverse tuning is dode.
    :type reverse: bool, optional

    :return dataset transformed
    :rtype class:`pandas.core.series.Series`

    """

    if reverse:
        return data * raw_data.apply(abs).sum(axis=0)
    else:
        return data / data.apply(abs).sum(axis=0)


def L2(data, raw_data=None, reverse=False):
    """

    Function for data normalization. Consists of the L2 normalization formula.
    value / sum(value) ^ 0.5

    :param data: data to process.
    :type data: class: `pandas.core.series.Series`
    :param raw_data: Raw data of the dataset. Only required for reverse transformation.
    :type raw_data: class:`pandas.core.series.Series`, optional
    :param reverse: Boolean parameter to determine if tuning of reverse tuning is done.
    :type reverse: bool, optional

    :return dataset transformed
    :rtype class:`pandas.core.series.Series`
    """

    if reverse:
        return data * (raw_data.sum(axis=0)) ** 0.5
    else:
        return data / (data.sum(axis=0)) ** 0.5


def std(data, mean_, std_, reverse=False):
    """
    Function for data standarisation. Consists of the Standaisation formula.
    (value - mean) / STD

    :param data: data process.
    :type data: class: `pandas.core.series.Series`
    :param mean_: Mean value of the dataset.
    :type mean_: class:`pandas.core.series.Series`
    :param std_: Standard Deviation value of the dataset.
    :type std_: class:`pandas.core.series.Series`
    :param reverse: Boolean parameter to determine if tuning of reverse tuning is dode.
    :type reverse: bool, optional

    :return dataset transformed
    :rtype class:`pandas.core.series.Series`

    """

    if reverse:
        return data * std_ + mean_
    else:
        return (data - mean_) / std_


__all__ = ["minmax", "L1", "L2", "std"]
