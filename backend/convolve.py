import time

import numpy as np
from numpy.lib.stride_tricks import as_strided

def convolve(image, conv_filter):
    filter_height, filter_width, in_channels = conv_filter.shape
    img_height, img_width, img_channels = image.shape

    assert in_channels == img_channels, "Filter and image must have the same number of channels."

    pad_h = (filter_height - 1) // 2
    pad_w = (filter_width - 1) // 2

    padded_image = np.pad(
        image,
        ((pad_h, pad_h), (pad_w, pad_w), (0, 0)),
        mode='constant'
    )

    # Use stride_tricks to create a view of the image with sliding windows
    # This avoids explicit loops and is much faster
    view_shape = (img_height, img_width, filter_height, filter_width, in_channels)
    strides = padded_image.strides[:2] + padded_image.strides
    sub_matrices = as_strided(padded_image, shape=view_shape, strides=strides)

    # Perform convolution using vectorized operations (einsum is great for this)
    # This computes the sum of the element-wise product of the windows and the filter
    # The output should be 2D (h, w) for a single filter application.
    output = np.einsum('h w H W c, H W c -> h w', sub_matrices, conv_filter)

    return output