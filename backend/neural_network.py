import numpy as np

def flatten_output(result):
    height, width, channels = result.shape
    flattened_shape = (height * width * channels)
    flattened_result = np.reshape(result, flattened_shape)
    return flattened_result

def relu_activation(result):
    return np.maximum(result, 0)

def hidden_dense_layer(input, weights, bias):
    output_before_activation = np.dot(input, weights) + bias
    activated_output = relu_activation(output_before_activation)
    return activated_output, output_before_activation

def final_dense_layer(input, weights, bias):
    output = np.dot(input, weights) + bias
    activated_output = sigmoid(output)
    return activated_output

def sigmoid(result):
    return 1 / (1 + np.exp(-result))

def binary_cross_entropy_loss(real, predicted):
    epsilon = 1e-15
    predicted = np.clip(predicted, epsilon, 1 - epsilon)
    loss = -np.mean(real * np.log(predicted) + (1 - real) * np.log(1 - predicted))
    return loss