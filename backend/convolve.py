import numpy as np

def convolve(image, conv_filter):
    filter_height, filter_width, in_channels = conv_filter.shape
    img_height, img_width, img_channels = image.shape

    pad_h = (filter_height - 1) // 2
    pad_w = (filter_width - 1) // 2

    # Pad the image with zeros on height and width, keep channels unchanged
    padded_image = np.pad(
        image,
        ((pad_h, filter_height - 1 - pad_h), (pad_w, filter_width - 1 - pad_w), (0, 0)),
        mode='constant'
    )

    output = np.zeros((img_height, img_width, in_channels))

    for c in range(in_channels):
        for i in range(img_height):
            for j in range(img_width):
                patch = padded_image[i:i+filter_height, j:j+filter_width, c]
                output[i, j, c] = np.sum(patch * conv_filter[:, :, c])
    return output