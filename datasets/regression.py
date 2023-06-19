"""
Hermes Dataset preprocessor for Regression
==========================================

This class processes numerical and categorical data for datasets used in regression, classification trees,
clustering models...

Author: Alvaro Marcos Canedo
"""

from hermes.data.base_dataset import BaseDataset

from hermes.data.base_data import NumericalData, CategoricalData

from hermes.utilities.enums import DatasetType, MissingData, Normalization, CategoricalData

from hermes.utilities.data_normalization import minmax, L1, L2, std, robust_scaler

import pandas as pd
from pandas.api.types import CategoricalDtype


class Regression(BaseDataset):
    """

    This class processed numerical and categorical data for regression, classification trees, clustering...

    """

    def __init__(self):
        """

        Constructor method.

        """
        super().__init__(DatasetType.REGRESSION)
        self.__numerical_data = {}
        self.__categorical_data = {}

    def prepare_data(self, input_data: pd.DataFrame, numerical_cols: list = None, categorical_cols: list = None,
                     categorical_type: hermes.utilities.enums.CategoricalData = CategoricalData.ONE_HOT,
                     normalization: hermes.utilities.enums.Normalization = Normalization.STD,
                     missing_data: hermes.utilities.MissingData = MissingData.MEAN) -> None:
        """

        Method to prepare the dataset for

        :param input_data: Input data for the dataset model.
        :type input_data: pd.DataFrame
        :param numerical_cols: List of columns to apply the numerical data preparation.
        :type numerical_cols: list
        :param categorical_cols: List of columns to apply the categorical data preparation.
        :type categorical_cols: list
        :param categorical_type: Categorical method to apply to categorical columns. Default is one hot encoder.
        :type categorical_type: hermes.utilities.enums.CategoricalData
        :param normalization: Normalization method to apply to the numerical columns. Default is based on standard dev.
        :type normalization: hermes.utilities.enums.Normalization
        :param missing_data: Missing data method to apply to numerical columns. Default is mean method.
        :type missing_data: hermes.utilities.enums.MissingData
        """

        model = {}

        # Set model info
        self.model.set_model_info(numerical_cols, categorical_cols, categorical_type, normalization, missing_data)

        # Numerical data preparation
        if numerical_cols:
            for i in numerical_cols:
                self.model.__numerical_data[i] = NumericalData(i, input_data[i])
                input_data[i] = self._missing_data(input_data[i], self.model.__numerical_data[i], missing_data)
                model[i] = self._parameter_tuning(input_data[i], self.model.__numerical_data[i], normalization)

        # Categorical data preparation
        if categorical_cols:
            for i in categorical_cols:
                self.model.__categorical_data[i] = CategoricalData(i, input_data[i])
                if categorical_type == CategoricalData.DUMMY:
                    model[i] = self._dummy_encoder(input_data[i], self.model.__categorical_data[i].encoder)
                elif categorical_type == CategoricalData.ONE_HOT:
                    model[i] = self._one_hot_encoder(input_data[i], self.model.__categorical_data[i].categories)

        # Concatenate all columns in a single DataFrame
        for key, value in model.items():
            self.model.data = pd.concat([self.model.data, value], axis=1)

    def revert_tuning(self, data: pd.Series, col_name: str, normalization: hermes.utilities.enums.Normalization) -> pd.Series:
        """
        Method to revert parameter tuning.

        :param data: Data to revert the transformation
        :type data: pd.Series
        :param col_name: Name of the column to grab the information.
        :type col_name: str
        :param normalization: Normalization/Standardisation technique to use.
        :type normalization: enums.Normalization

        :return: Reverted Tuned data.
        :rtype: pd.Series
        """
        return self._parameter_tuning(data, self.numerical_data[col_name], normalization, reverse=True)

    @staticmethod
    def _dummy_encoder(data: pd.Series, encoder: dict) -> pd.Series:
        """
        Encode data as a dummy integer ordinal numbers.

        :param data: Series of values.
        :type data: pd.Series
        :param encoder: List of categories.
        :type encoder: dict

        :return Data encoded as dummy variables.
        :rtype pd.Series
        """
        for key, value in encoder.items():
            data[data == key] = value
        return data

    @staticmethod
    def _one_hot_encoder(data: pd.Series, categories: list) -> pd.Series:
        """
        Encode data as a one-hot binary columns.
        :param data: Series of values.
        :type data: pandas.Series
        :param categories: List of categories.
        :type categories: list

        :return One hot encoded data
        :rtype pd.Series
        """
        data = data.astype(CategoricalDtype(categories=categories, ordered=True))
        return pd.get_dummies(data, prefix=data.name)

    @staticmethod
    def _parameter_tuning(data: pd.Series, col_data: hermes.data.base_data.NumericalData,
                          normalization: hermes.utilities.enums.Normalization, reverse: bool = False) -> pd.Series:
        """
        Method for tuning the parameters for the different calculations.
        :param data: database values.
        :type data: pd.Series
        :param col_data: column parameter information.
        :type col_data: hermes.utilities.base_data.NumericalData
        :param normalization: Type of normalization to apply to the database.
        :type normalization: hermes.utilities.enums.Normalization
        :param reverse: Parameter to apply reverse tuning to the data.
        :type reverse: bool
        :return Updated database (normalized or standardized)
        :rtype pd.Series
        """
        if normalization == Normalization.MINMAX:
            return minmax(data, col_data.min_, col_data.max_, reverse)
        elif normalization == Normalization.L1:
            return L1(data, reverse)
        elif normalization == Normalization.L2:
            return L2(data, reverse)
        elif normalization == Normalization.STD:
            return std(data, col_data.mean_, col_data.std_, reverse)
        elif normalization == Normalization.ROBUST_SCALER:
            return robust_scaler(data, col_data.median_, col_data.quantile75 - col_data.quantile25)
        else:
            return data

    @staticmethod
    def _missing_data(data: pd.Series, key: hermes.data.base_data,
                      missing_data: hermes.utilities.enums.MissingData) -> pd.Series:
        """
        Method to work with missing data from the input data.
        :param data: input data of the dataset.
        :type data: pd.Series
        :param key: Information about the column of the DataFrame.
        :type key: hermes.data.base_data
        :param missing_data: Form of replacing missing data from the model.
        :type missing_data: hermes.utilities.enums.MissingData

        :return  Dataframe with missing data problem solved.
        :rtype pd.Series
        """

        if missing_data == enums.MissingData.MEAN:
            return data.fillna(key.mean_)
        elif missing_data == enums.MissingData.STD:
            return data.fillna(key.std_)
        elif missing_data == enums.MissingData.MAX:
            return data.fillna(key.max_)
        elif missing_data == enums.MissingData.MIN:
            return data.fillna(key.min_)
        elif missing_data == enums.MissingData.DELETE:
            return data.dropna(how='all')
        elif missing_data == enums.MissingData.MODE:
            return data.fillna(key.mode_)
        else:
            return data


__all__ = ["regression"]
