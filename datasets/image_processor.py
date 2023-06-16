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

import os


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
                     normalization_method: Normalzation = Normalization.RANGE_NORM, range_: tuple = (0, 1)):
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

    @staticmethod
    def _get_normalized_image(normalization_type: Normalization, image: Image, range_: tuple = (0, 1)) -> Image:
        """

        Static method to get the normalization type for the images.

        :param normalization_type: Normalization type to apply
        :type normalization_type: Normalization
        :param image: Image to scale
        :type image: Image
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
