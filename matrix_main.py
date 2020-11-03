import numpy as np


def sigmoid(matrix):
    return 1.0 / (1.0 + np.exp(-matrix))


if __name__ == '__main__':
    input_nodes = 8
    hidden_nodes = 3
    output_nodes = 8

    # Weight initialization
    theta1 = np.random.random((input_nodes + 1, hidden_nodes))
    theta2 = np.random.random((hidden_nodes + 1, output_nodes))

    train_examples = 3
    x = np.zeros((8, train_examples))

    # Add the '1' activation of the bias
    bias_x = np.ones((1, train_examples))
    x = np.concatenate((x, bias_x))

    # Multiply the input by the weights
    a2 = np.dot(theta1.transpose(), x)
    a2 = sigmoid(a2)

    # Add the output of 1 for the bias again
    bias_a2 = np.ones((1, a2.shape[1]))
    a2 = np.concatenate((a2, bias_a2))

    a3 = np.dot(theta2.transpose(), a2)
    a3 = sigmoid(a3)

    print(a3)