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


# Forward propagation input -> output
def forward_propagation(network, row):
    inputs = row
    for layer in network:  # For each layer in the network
        new_inputs = None
        for neuron in layer: # For each separate 'neuron' in the network
            weights = neuron['weights'][0]
            activation = activate(weights, inputs)  # multiply the corresponding weights with the input
            neuron['outputs'] = sigmoid(activation) # sigmoid function to map it between 0 and 1

            # The 'outputs' of the current layer are saved to the new_inputs
            if new_inputs is None:
                new_inputs = neuron['outputs']
            else:
                new_inputs = np.concatenate((new_inputs, neuron['outputs']), axis=1)
        inputs = new_inputs
    # And we return the network output
    return inputs


# Make the training vectors, the number will be the amount of examples used
def make_training(number):
    train = np.zeros((number, 8))
    for i in range(number):
        index = np.random.randint(0, 8)
        train[i][index] = 1
    return train


def main():
    # Make the network, 8 input, 3 hidden, 8 output
    network = make_network(8, 3, 8)

    # Make the training examples
    train = make_training(6)

    # The first output of the network with random weights
    output = forward_propagation(network, train)


if __name__ == '__main__':
    main()
