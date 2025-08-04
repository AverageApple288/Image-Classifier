import cupy as cp
import numpy as np

def generate_layer(filter_height, filter_width, in_channels, out_channels):
    # fan_in is the number of connections contributing to the output of a neuron in this layer.
    # For a conv layer, it's filter_height * filter_width * in_channels
    fan_in = filter_height * filter_width * in_channels
    std_dev = cp.sqrt(2.0 / fan_in)

    # Draw from a normal distribution with mean 0 and calculated std_dev
    filters = cp.random.normal(loc=0.0, scale=std_dev, size=(filter_height, filter_width, in_channels, out_channels))
    return filters