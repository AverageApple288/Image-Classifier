import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from backend.convolve import convolve
from backend.max_pool import max_pool


def process_image(image_path, layers, target_size=(500,500)):
    """Processes a single image through all layers."""
    image = Image.open(image_path).convert('RGB')
    image = image.resize(target_size)

    pixelinfo = np.array(image, dtype=np.float32) / 255.0

    if len(pixelinfo.shape) == 2:
        # Ensure 3 channels for grayscale images, as filters expect this
        pixelinfo = np.stack((pixelinfo,) * 3, axis=-1)

    all_layer_outputs = []
    current_input = pixelinfo

    # Process all layers
    for i, layer_filters in enumerate(layers):
        out_channels = layer_filters.shape[3]
        output_maps_for_layer = []

        for j in range(out_channels):
            # Each filter is applied to the full input from the previous layer
            filter_slice = layer_filters[:, :, :, j]

            # Convolve and apply ReLU activation
            convolved_map = convolve(current_input, filter_slice)
            activated_map = np.maximum(0, convolved_map)

            # Pool the result
            # Add a temporary channel axis for the max_pool function
            pooled_map = max_pool(activated_map[:, :, np.newaxis], pool_size=2, stride=2)
            output_maps_for_layer.append(pooled_map)

        all_layer_outputs.append(output_maps_for_layer)

        # The concatenated output of this layer is the input for the next
        if output_maps_for_layer:
            current_input = np.concatenate(output_maps_for_layer, axis=-1)

    # Return the feature maps for all layers for the image
    return all_layer_outputs

"""
    image = Image.open(image_path).convert('RGB')
    image = image.resize(target_size)

    pixelinfo = np.array(image, dtype=np.float32) / 255.0

    if len(pixelinfo.shape) == 2:
        pixelinfo = np.expand_dims(pixelinfo, axis=-1)

    normal_maps = [[] for _ in range(len(layers))]

    # Layer 0
    out_channels_layer0 = layers[0].shape[3]
    for j in range(out_channels_layer0):
        filter_slice = layers[0][:, :, :, j]
        normal_map = convolve(pixelinfo, filter_slice) # Output is 2D
        normal_map = np.maximum(0, normal_map)
        # Add channel dimension for max_pool
        normal_map = max_pool(normal_map[:, :, np.newaxis], pool_size=2, stride=2)
        normal_maps[0].append(normal_map)

    # Subsequent layers
    for i in range(1, len(layers)):
        # Concatenate along the channel axis
        input_maps = np.concatenate(normal_maps[i-1], axis=-1)
        out_channels = layers[i].shape[3]
        for j in range(out_channels):
            filter_slice = layers[i][:, :, :, j]
            normal_map = convolve(input_maps, filter_slice) # Output is 2D
            normal_map = np.maximum(0, normal_map)
            # Add channel dimension for max_pool
            normal_map = max_pool(normal_map[:, :, np.newaxis], pool_size=2, stride=2)
            normal_maps[i].append(normal_map)

    # Return the final feature maps for the image
    return normal_maps
"""