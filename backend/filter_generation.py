import numpy as np

def generate_filter(filter_height, filter_width, in_channels, out_channels):
    # fan_in is the number of connections contributing to the output of a neuron in this layer.
    # For a conv layer, it's filter_height * filter_width * in_channels
    fan_in = filter_height * filter_width * in_channels
    std_dev = np.sqrt(2.0 / fan_in)

    # Draw from a normal distribution with mean 0 and calculated std_dev
    filters = np.random.normal(loc=0.0, scale=std_dev, size=(filter_height, filter_width, in_channels, out_channels))
    return filters