"""
Hermes image normalization function module
===========================================

This file contains the function related to the normalization of the images of a database.

Author: Alvaro Marcos Canedo
"""

from PIL import Image

import numpy as np


def range_normalization(image: Image) -> Image:
    """

    Range normalization technique for images. It is normalized to 255.0, giving a range of pixels between 0 and 1.

    :param image: Image to normalize.
    :type image: Image
    :return: Normalized image for the database.
    :rtype: Image
    """

    # Convert image to floating point
    image = image.convert("F")

    # Return normalized image
    return image.point(lambda x: x / 255.0)


def minmax(image: Image, range_: tuple) -> Image:
    """

    Min Max scaler for images. At the moment, it is done with the minimum and maximum value of each image.

    :param image: Image to be scaled.
    :type image: Image
    :param range_: Range to scale the image. Must be between 0 and 1, or -1 and 1.
    :type range_: tuple
    :return: Scaled image for the database.
    :rtype: Image
    """
    # TODO -> Implement minmax scaler with the global min and max value of the entire dataset.

    # Convert image to numpy array
    image_array = np.array(image)

    # Get min and max value
    min_value = np.min(image_array)
    max_value = np.max(image_array)

    # Scale numpy image
    scaled_image_array = ((image_array - min_value) / (max_value - min_value) * (range_[1] - range_[0]) + range_[0]).astype(
        np.uint8)

    # Return of the scaled image
    return Image.fromarray(scaled_image_array)


def channel_wise(image: Image) -> Image:
    """

    Channel wise normalization technique It scales each channel of the image so all are scaled. It may improve deep
    learning models performance.

    :param image: Image to scale.
    :type image: Image
    :return: Scaled image for the computer vision model.
    :rtype: Image
    """
    # Convert image to numpy array
    image_array = np.array(image)

    # Compute Standard deviation and mean of each channel of the image
    channels_mean = np.mean(image_array, axis=(0, 1))
    channels_std = np.sdt(image_array, axis=(0, 1))

    # Scale array image
    image_array = (image_array - channels_mean) / channels_std

    return Image.fromarray(image_array.astype(np.uint8))


__all__ = ["range_normalization", "minmax"]
