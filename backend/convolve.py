import cupy as cp

def convolve(image, conv_filter):
    """Efficiently convolve an image with a filter using GPU parallelization.

    Args:
        image: Input image tensor of shape (height, width, channels)
        conv_filter: Convolution filter of shape (filter_height, filter_width, channels)

    Returns:
        Convolved output of shape (height, width, channels)
    """
    filter_height, filter_width, in_channels = conv_filter.shape
    img_height, img_width, img_channels = image.shape

    pad_h = (filter_height - 1) // 2
    pad_w = (filter_width - 1) // 2

    # Pad the image with zeros on height and width, keep channels unchanged
    padded_image = cp.pad(
        image,
        ((pad_h, filter_height - 1 - pad_h), (pad_w, filter_width - 1 - pad_w), (0, 0)),
        mode='constant'
    )

    # Prepare output array
    output = cp.zeros((img_height, img_width, in_channels))

    # Use GPU-accelerated batch processing for each channel
    for c in range(in_channels):
        # Create a view of the padded image for the current channel
        padded_channel = padded_image[:, :, c]
        filter_channel = conv_filter[:, :, c]

        # Use CuPy's correlate2d equivalent - we build it manually for maximum GPU utilization
        # This creates a kernel that applies the filter to all positions at once
        for i in range(filter_height):
            for j in range(filter_width):
                output[:, :, c] += padded_channel[i:i+img_height, j:j+img_width] * filter_channel[i, j]

    return output