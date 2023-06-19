"""

Hermes Image processor
========================

This class serves as an image processor for Computer vision models.

Author: Alvaro Marcos Canedo
"""

from hermes.data.base_dataset import BaseDataset
from hermes.utilities.enums import DatasetType, Normalization
from hermes.utilities.image_normalization import range_normalization, minmax, channel_wise

from PIL import Image
from path import Path
from shutil import copyfile
from random import sample

import os
import tensorflow as tf


class ImageProcessor(BaseDataset):
    """

    This class is an image processor for computer vision models.

    """

    def __init__(self):
        """

        Constructor method.

        """
        super().__init__(DatasetType.IMAGE_PROCESSOR)

    def prepare_data(self, file_path: str, image_size: tuple = None,
                     normalization_method: Normalzation = Normalization.RANGE_NORM, range_: tuple = (0, 1),
                     split_size: float = 1.0) -> None:
        """

        Method to prepare the image dataset for computer vision models.
        It is supposed that all images that will be part of the computer vision model are stored in folders which name
        are descriptive enough. So, only a path to the main directory of the database is required, Hermes image
        processor will do the rest. It will create a subdirectory containing the processed images, preserving the
        location and name of each one.

        :param file_path: Path to the main directory of the image database.
        :type file_path: str
        :param image_size: Desired image size for the database. If not provided, not applied.
        :type image_size: tuple, optional
        :param normalization_method: Normalization method to apply to the images.
        :type normalization_method: Normalization, optional
        :param range_: Range to normalize the images. Only used if the normalization method is set to MinMax.
        :type range_: tuple, optional
        :param split_size: Proportion of dataset to be used for training. Default -> 1.0 -> All images for training
        :type split_size: float, optional
        :return: None return for this method.
        :rtype: None
        """

        # Set model info for the processor.
        self.model.set_path(file_path)

        # Loop over images and apply every change
        database_list = self.model.path.generator_path.listdir()

        # Iterate over folders to resize each image. It will be resized and copied to another folder.
        for folder in database_list:

            # Set folder to store images
            folder_path = os.path.join(self.model.path.generator_path, folder)

            # List all available images in the folder
            image_list = Path.listdir(folder_path)

            # Output folder path
            output_folder = os.path.join(self.model.path.output_path, folder)
            if not os.path.exists(output_folder):
                os.mkdir(output_folder)

            # Loop over sub folders to access every image
            for image in image_list:
                image_path = os.path.join(folder_path, image)
                with Image.open(image_path) as image_:

                    # Resize the image only if the size is provided in the input
                    if image_size:
                        image_ = image_.resize(image_size)

                    # Normalization of image based on input provided
                    image_ = self._get_normalized_image(normalization_method, image_, range_)

                    # Save image for the computer visio model
                    image_.save(os.path.join(output_folder, image))

            # Split data into validation and training directories
            if split_size:
                self._split_data(output_folder, split_size)

                # Delete source folder
                os.rmdir(output_folder)
            else:

                self.model.path.training_path = self.model.path.output_path
                delattr(self.model.path, 'validation_path')

    def create_model_generators(self, batch_size: int = 32, label_mode: str = 'int', color_mode: str = 'rgb',
                                shuffle: bool = True) -> None:
        """

        Method to create generator for training and validation folders for computer vision models

        :param batch_size: Size of the batches of data. Default: 32
        :type batch_size: int
        :param label_mode: Description of encoding labels. Default: 'int'
        :type label_mode: str
        :param color_mode: Color mode for the model. Default: 'rgb'
        :type color_mode: str
        :param shuffle: Whether to shuffle the data. Default: True
        :type shuffle: boll
        :return:
        """

        # Call preprocessing method image_dataset_from_directory for training generator
        self.model.nn_data.training_gen = tf.keras.preprocessing.image_dataset_from_directory(
            self.model.path.training_path,
            image_size=self.model.info.image_size,
            batch_size=batch_size,
            labels='inferred',
            label_mode=label_mode,
            color_mode=color_mode,
            shuffle=shuffle
        )

        # Call flow_from_directory method
        self.model.nn_data.validation_gen = tf.keras.preprocessing.image_dataset_from_directory(
            self.model.path.validation_path,
            image_size=self.model.info.image_size,
            batch_size=batch_size,
            labels='inferred',
            label_mode=label_mode,
            color_mode=color_mode
            )

    def preprocessing_layers_for_nn(self, random_flip: str = None, random_rotation: float = None,
                                    height_shift: tuple = None, width_shift: tuple = None,
                                    height_zoom: tuple = None, width_zoom: tuple = None):

        model = tf.keras.Sequential([])

        # Add random flip preprocessing layer
        if random_flip:
            model.add(tf.keras.layers.RandomFlip(mode=random_flip))

        # Add random rotation preprocessing layer
        if random_rotation:
            model.add(tf.keras.layers.RandomRotation(random_rotation=random_rotation))

        # Add random translation preprocessing layer
        if height_shift or width_shift:
            model.add(tf.keras.layers.RandomTranslation(height_shift, width_shift))

        # Add random zoom preprocessing layer
        if height_zoom or width_zoom:
            model.add(tf.keras.layers.RandomZoom(height_zoom, width_zoom))

        # Copy the preprocessing layers to the attribute of the class
        self.model.nn_data.preprocessing_layers = model

    def _split_data(self, images_path: Path, split_size: float):
        """

        Method to split image data into training and validation directory.

        :param images_path: Path to the folder where images are stored.
        :type images_path: Path
        :param split_size: Portion of the dataset to be used for training. The rest will be used for the validation.
        :type split_size: float
        :return:
        """

        # Get full list of images
        source_list = images_path.listdir()

        # Generate random sample for training folder (the rest will be validation)
        random_sample = sample(source_list, int(split_size * len(source_list)))

        # Loop over list to move images from source directory to training and validation folders
        for image_path in source_list:
            # Check if file is in the random sample generated
            if image_path in random_sample:
                # If true, goes to training path
                copyfile(image_path, os.path.join(self.model.path.training_path, images_path.stem))
            else:
                # Else, copies the file to the validation folder
                copyfile(image_path, os.path.join(self.model.path.validation_path, images_path.stem))

    @staticmethod
    def _get_normalized_image(normalization_type: Normalization, image: Image, range_: tuple = (0, 1)) -> Image:

        """

        Static method to get the normalization type for the images.

        :param normalization_type: Normalization type to apply
        :type normalization_type: Normalization
        :param image: Image to scale
        :type image: Image
        :param range_: Range to apply to image normalization.
        :type range_: tuple, optional
        :return: Normalized/Scaled/Standardized image.
        :rtype: Image
        """

        if normalization_type == Normalization.MINMAX:
            return minmax(image)
        elif normalization_type == Normalization.RANGE_NORM:
            return range_normalization(image, range_)
        elif normalization_type == Normalization.CHANNEL_WISE:
            return channel_wise(image)
        else:
            raise NotImplementedError("Normalization method not implemented at the moment")


__all__ = ["ImageProcessor"]
