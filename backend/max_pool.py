import numpy as np
from numpy.lib.stride_tricks import as_strided

def max_pool(image, pool_size=2, stride=2):
    img_height, img_width, channels = image.shape
    out_height = (img_height - pool_size) // stride + 1
    out_width = (img_width - pool_size) // stride + 1

    # Create a view of the image with sliding windows
    view_shape = (out_height, out_width, channels, pool_size, pool_size)
    
    # Calculate strides for the view
    # Stride for height, width, channels, then inside the pool window
    strides = (
        stride * image.strides[0], 
        stride * image.strides[1], 
        image.strides[2], 
        image.strides[0], 
        image.strides[1]
    )
    
    windows = as_strided(image, shape=view_shape, strides=strides)

    # Find the maximum value in each window across the pool_size x pool_size dimensions
    # The axes to reduce are the last two (pool_height and pool_width)
    pooled = np.max(windows, axis=(3, 4))
    
    return pooled