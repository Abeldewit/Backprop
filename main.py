from math import exp
from random import random
import numpy as np


# Make the appropriate layers with randomized weights
def make_network(input_nodes, hidden_nodes, output_nodes):
    network = list()

    # Create the hidden layer with 'hidden_nodes' amount of nodes,
    # and connected to 'input_nodes' amount of nodes + 1 for the bias
    hidden_layer = [{'weights': [np.random.random((input_nodes + 1, 1))]} for _ in range(hidden_nodes)]
    network.append(hidden_layer)

    # Create the output layer with 'output_nodes' amount of nodes,
    # and connected to 'hidden_nodes' amount of nodes + 1 for the bias
    output_layer = [{'weights': [np.random.random((hidden_nodes + 1, 1))]} for _ in range(output_nodes)]
    network.append(output_layer)

    return network


# The activation (simply multiply the weights with the inputs)
def activate(weights, inputs):
    activation = float(weights[-1])  # The bias

    # Delete the bias
    node_weights = np.delete(weights, -1, 0)

    # Dot product of the input and weights
    dot = np.dot(inputs, node_weights)

    # Add this up to the bias
    activation += dot
    return activation


# The Sigmoid function
def sigmoid(activation):
    return 1.0 / (1.0 + np.exp(-activation))


# The derivative of the Sigmoid
def sigmoid_derivative(output):
    return output * (1.0 - output)


def forward_propagation(network, row):
    inputs = row
    for layer in network:
        new_inputs = None
        for neuron in layer:
            weights = neuron['weights'][0]
            activation = activate(weights, inputs)
            neuron['outputs'] = sigmoid(activation)
            if new_inputs is None:
                new_inputs = neuron['outputs']
            else:
                new_inputs = np.concatenate((new_inputs, neuron['outputs']), axis=1)
        inputs = np.array(new_inputs)
    return inputs


def make_training(number):
    train = np.zeros((number, 8))
    for i in range(number):
        index = np.random.randint(0, 8)
        train[i][index] = 1
    return train


def main():
    network = make_network(8, 3, 8)
    train = make_training(6)

    output = forward_propagation(network, train)


if __name__ == '__main__':
    main()
