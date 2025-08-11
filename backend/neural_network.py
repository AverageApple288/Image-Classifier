import numpy as np

def flatten_output(result):
    height, width, channels = result.shape
    flattened_shape = (height * width * channels)
    flattened_result = np.reshape(result, flattened_shape)
    return flattened_result

def relu_activation(result):
    return np.maximum(result, 0)

def dense_layer(input, weights, bias):
    output = np.dot(input, weights) + bias
    activated_output = relu_activation(output)
    return activated_output

def final_dense_layer(input, weights, bias):
    output = np.dot(input, weights) + bias
    activated_output = sigmoid(output)
    return activated_output

def sigmoid(result):
    return 1 / (1 + np.exp(-result))