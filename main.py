from math import exp
from random import random
import numpy as np
import time

from NNdraw import DrawNN


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
            neuron['outputs'] = sigmoid(activation)  # sigmoid function to map it between 0 and 1

            # The 'outputs' of the current layer are saved to the new_inputs
            if new_inputs is None:
                new_inputs = neuron['outputs']
            else:
                new_inputs = np.concatenate((new_inputs, neuron['outputs']), axis=1)
        inputs = new_inputs
    # And we return the network output
    return inputs


# Backward propagation output -> weight update
def backward_propagation(network, expected, output):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = []
        if i == len(network)-1:  # Calculate the error for the output layer (len -1) is the last layer
            for j, neuron in enumerate(layer):
                errors.append(expected[0][j] - neuron['outputs'])
        else:  # Else we calculate the error for the hidden/interior nodes
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][0][j] * neuron['delta'])
                errors.append(error)
        for j, neuron in enumerate(layer):
            neuron['delta'] = errors[j] * sigmoid_derivative(neuron['outputs'])


def update_weights(network, row, learning_rate):
    for i in range(len(network)):
        inputs = row[0]
        if i != 0:
            inputs = [neuron['outputs'][0] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                update = learning_rate * inputs[j] * neuron['delta'][0]
                neuron['weights'][0][j] += update

            bias_update = learning_rate * neuron['delta'][0]
            neuron['weights'][0][-1] += bias_update


# Make the training vectors, the number will be the amount of examples used
def make_training(number):
    train = np.zeros((8, number))
    for i in range(number):
        index = np.random.randint(0, 8)
        train[index][i] = 1
    return train


def make_all():
    train = np.zeros((8, 8))
    for i in range(8):
        train[i][i] = 1
    return train


def main():
    # Make the network, 8 input, 3 hidden, 8 output
    network = make_network(8, 3, 8)

    # Make the training examples
    training = make_all()
    training = training.transpose()

    GS = input('GridSearch (y/n): ')

    if GS.lower() == 'n':
        learn_rate = 0.5
        n_epochs = 1000

        for epoch in range(n_epochs):
            sum_error = 0
            for i in range(training.shape[0]):
                train = [training[i]]
                output = forward_propagation(network, train)

                dif = train - output
                sum_dif = np.sum(np.square(dif))
                sum_error += sum_dif

                backward_propagation(network, train, output)
                update_weights(network, train, learn_rate)

            print('>epoch {}\t lr = {}, error = {}'.format(epoch+1, learn_rate, round(sum_error, 3)))
    elif GS.lower() == 'y':

        rates = [0.1, 0.25, 0.5, 0.75, 0.9]
        eps = [10, 100, 1000, 2000, 5000]

        errors = []
        for r in range(len(rates)):
            for j in range(len(eps)):
                learn_rate = rates[r]
                n_epochs = eps[j]

                start_time = time.time()
                for epoch in range(n_epochs):
                    sum_error = 0
                    for i in range(training.shape[0]):
                        train = [training[i]]
                        output = forward_propagation(network, train)

                        dif = train - output
                        sum_dif = np.sum(np.square(dif))
                        sum_error += sum_dif

                        backward_propagation(network, train, output)
                        update_weights(network, train, learn_rate)
                elapsed = time.time() - start_time
                print((learn_rate, n_epochs, sum_error, elapsed))
                errors.append((learn_rate, n_epochs, sum_error, elapsed))
            print('-'*20)

        best_score = (0, 0, float('inf'), 0)

        best_time = (0, 0, 0, float('inf'))
        for er in errors:
            if er[2] < best_score[2]:
                best_score = er
            if er[3] < best_time[3]:
                best_time = er

        print(best_score)
        print(best_time)


    print('\n')

    while True:
        test = input('test: ')

        if test.lower() == 'vis':
            drawer = DrawNN([8, 3, 8], network)
            drawer.draw()
            continue

        if test.lower() == 'print':
            for layer in network:
                for neuron in layer:
                    print('w:', neuron['weights'][0][:-1])
                    print('b:', neuron['weights'][0][-1])
            continue

        if test.lower() == 'exit':
            break

        t = []
        for c in test:
            t.append(int(c))
        if len(t) == 8:
            output = np.round(forward_propagation(network, [t]))[0]
            print(output, end='\t')

            correct = True
            for i, e in enumerate(t):
                if e != output[i]:
                    correct = False
            if correct:
                print('CORRECT!')
            else:
                print('wrong :(')


if __name__ == '__main__':
    main()
