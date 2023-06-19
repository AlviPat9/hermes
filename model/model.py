"""
Hermes model class
====================

This class describes the model information for each data preprocessor.

Author: Alvaro Marcos Canedo
"""

import pandas as pd
import os

from path import Path

from hermes.utilities.enums import CategoricalData, Normalization, MissingData


class RegModel(object):
    """

    This is the class describing the information about the model for the regression data preprocessor. Sets the
    DataFrame for the model conversion and other additional information.

    """

    def __init__(self):
        """

        Constructor method.

        """

        # Output data for the
        self.data = pd.DataFrame()
        self.__numerical_data = {}
        self.__categorical_data = {}
        self.info = None

    def set_model_info(self, numerical_columns: list, categorical_columns: list,
                       categorical_conversion: hermes.utilities.enums.Categoricaldata,
                       normalization_method: hermes.utilities.enums.Normalization,
                       missing_data_method: hermes.utilities.enums.MissingData) -> None:
        """

        Method to set the model info for the user.

        """

        self.info.numerical_columns = numerical_columns
        self.info.categorical_columns = categorical_columns

        # Set categorical treatment type
        if categorical_conversion == CategoricalData.ONE_HOT:
            self.info.categorical_conversion = "One-Hot encoder"
        elif categorical_conversion == CategoricalData.DUMMY:
            self.info.categorical_conversion = "Dummy encoder"

        # Set normalization method
        if normalization_method == Normalization.STD:
            self.info.normalization_method = "Standard deviation normalization"
        elif normalization_method == Normalization.MINMAX:
            self.info.normalization_method = "MinMax normalization"
        elif normalization_method == Normalization.L1:
            self.info.normalization_method = "L1 Regularization method"
        elif normalization_method == Normalization.L2:
            self.info.normalization_method = "L2 Regularization method"
        elif normalization_method == Normalization.ROBUST_SCALER:
            self.info.normalization_method = "Robust scaler method"

        # Set missing data method
        if missing_data_method == MissingData.MEAN:
            self.info.missing_data = "Missing values filled with MEAN"
        elif missing_data_method == MissingData.DELETE:
            self.info.missing_data = "Deleted rows with missing values"
        elif missing_data_method == MissingData.STD:
            self.info.missing_data = "Missing values filled with STANDARD DEVIATION"
        elif missing_data_method == MissingData.MIN:
            self.info.missing_data = "Missing values filled with MIN value"
        elif missing_data_method == MissingData.MAX:
            self.info.missing_data = "Missing values filled with MAX value"
        elif missing_data_method == MissingData.MODE:
            self.info.missing_data = "Missing values filled with MODE value"


class ImageModel(object):
    """

    This is the class describing the information about the model for the computer vision data preprocessor (Images in
    this case). Sets output paths and folders

    """

    def __init__(self):
        """

        Constructor method.

        """
        # Set model for the info
        self.info = "Preprocessor for Computer vision models"

        # Paths initialization
        self.path.generator_path = None
        self.path.output_path = None

        # Set Neural network input data
        self.nn_data.training_gen = None
        self.nn_data.validation_gen = None
        self.nn_data.preprocessing_layers = None

    def set_path(self, input_path: str):
        """

        Set input and output paths

        :param input_path: Path to the folders where images are stored.
        :type input_path: str
        :return:
        """

        # Set generator path
        setattr(self.path, 'generator_path', Path(input_path))

        # Set output path and chek if it exists
        setattr(self.path, 'output_path', Path(os.path.join(input_path, 'model')))

        # Check existence of the output path
        if not self.path.output_path.exists():
            Path.mkdir(self.path.output_path)

        # Set training path
        setattr(self.path, 'training_path', Path(os.path.join(self.path.output_path, 'training')))

        # Check existence of the training path
        if not self.path.training_path.exists():
            Path.mkdir(self.path.training_path)

        # Set validation path
        setattr(self.path, 'validation_path', Path(os.path.join(self.path.output_path, 'validation')))

        # Check existence of the validation path
        if not self.path.validation_path.exists():
            Path.mkdir(self.path.validation_path)

    def set_model_info(self):
        """

        Set information of the model.

        :return:
        """

        pass


__all__ = ["RegModel", "ImageModel"]
