import cupy as cp
from cupyx.scipy import signal

def convolve_optimized(image, conv_filter):
    """Fully optimized GPU-based convolution using CuPy's built-in functions.

    This implementation uses cupyx.scipy.signal for maximum GPU utilization.

    Args:
        image: Input image tensor of shape (height, width, channels)
        conv_filter: Convolution filter of shape (filter_height, filter_width, channels)

    Returns:
        Convolved output of shape (height, width, channels)
    """
    img_height, img_width, img_channels = image.shape
    filter_height, filter_width, in_channels = conv_filter.shape

    # Prepare output array
    output = cp.zeros((img_height, img_width, in_channels))

    # Process each channel using built-in GPU-optimized correlation
    for c in range(in_channels):
        output[:, :, c] = signal.correlate2d(
            image[:, :, c],
            conv_filter[:, :, c],
            mode='same',
            boundary='fill',
            fillvalue=0
        )

    return output


def batch_convolve(images, conv_filter):
    """Process a batch of images in parallel on the GPU.

    Args:
        images: Batch of images with shape (batch_size, height, width, channels)
        conv_filter: Convolution filter of shape (filter_height, filter_width, channels)

    Returns:
        Batch of convolved outputs with shape (batch_size, height, width, channels)
    """
    batch_size, img_height, img_width, img_channels = images.shape
    filter_height, filter_width, in_channels = conv_filter.shape

    # Prepare output array
    output = cp.zeros((batch_size, img_height, img_width, in_channels))

    # Process each channel for all images in the batch simultaneously
    for c in range(in_channels):
        # Reshape for batch processing
        batch_channel = images[:, :, :, c].reshape(batch_size, 1, img_height, img_width)
        kernel = conv_filter[:, :, c].reshape(1, 1, filter_height, filter_width)

        # Use CuPy's 2D convolution for the entire batch at once
        result = cp.cudnn.convolution_forward(
            batch_channel.astype(cp.float32),
            kernel.astype(cp.float32),
            pad=((filter_height-1)//2, (filter_width-1)//2),
            stride=(1, 1)
        )

        # Reshape result back to original dimensions
        output[:, :, :, c] = result.reshape(batch_size, img_height, img_width)

    return output
