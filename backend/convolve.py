import numpy as np

def convolve(image, conv_filter):
    filter_height, filter_width, in_channels = conv_filter.shape
    img_height, img_width, img_channels = image.shape

    out_height = img_height - filter_height + 1
    out_width = img_width - filter_width + 1
    output = np.zeros((out_height, out_width, in_channels))

    for c in range(in_channels):
        for i in range(out_height):
            for j in range(out_width):
                patch = image[i:i+filter_height, j:j+filter_width, c]
                output[i, j, c] = np.sum(patch * conv_filter[:, :, c])
    return output