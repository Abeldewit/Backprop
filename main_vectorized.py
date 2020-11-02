import numpy as np
import matplotlib.pyplot as plt
import time


# activation of one unit
def activate(weights, x):
    return 1 / (1 + np.exp(-np.dot(weights, x)))


# calculate the derivative of a single node output
def activation_derivative(activation):
    return activation * (np.ones(len(activation)) - activation)


# forward propagation to compute activations, returns input + activations
def forward_propagation(NN, x):
    activations = [x]
    for layer in NN:
        activations.append([])
        activations[-1] = activate(layer['weights'], np.concatenate(([1], activations[-2])))

    return activations


# perform backward propagation
def backward_propagation(NN, deltas, x, y):
    activations = forward_propagation(NN, x)

    # compute the 'errors' of all nodes
    weights_without_bias_output = NN[-1]['weights'][:, 1:]
    errors = [0, activation_derivative(activations[-1]) * (activations[-1] - y)]

    # print(activations[-2])
    # exit()

    errors[0] = activation_derivative(activations[-2]) * (np.dot(weights_without_bias_output.T, errors[-1]))

    # update the deltas using partial derivatives of the weights/biases
    for j in range(len(NN)):
        # print(errors[j])
        # print(activations[j])
        partial_derivative = np.c_[errors[j], np.outer(errors[j], activations[j])]
    #     print(partial_derivative)
    #
        deltas[j] += partial_derivative
    #
    # exit()

    return np.sum(np.square(errors[-1]))


# update the weights
def weight_update(NN, deltas, n_inputs, learning_rate, weight_decay):
    n_inputs = 1
    for i in range(len(NN)):
        # NN[i]['weights'] -= learning_rate * (1/n_inputs * deltas[i])
        # NN[i]['weights'][:, 1:] -= learning_rate * weight_decay * NN[i]['weights'][:, 1:]

        NN[i]['weights'][:, 1:] -= learning_rate * ((1 / n_inputs) * deltas[i][:, 1:] + weight_decay * NN[i][
            'weights'][:, 1:])
        NN[i]['weights'][:, 0] -= learning_rate * deltas[i][:, 0] / n_inputs


# performs gradient descent to learn the weights, using backward propagation, returns the summed error per iteration
def gradient_descent(NN, x, y, iter, learning_rate, weight_decay):
    learning_curve = []

    for i in range(iter):
        learning_curve.append(0)

        for j in range(len(x)):
            deltas = [np.zeros(layer['weights'].shape) for layer in NN]

            error = backward_propagation(NN, deltas, x[j], y[j])
            learning_curve[-1] += error

            weight_update(NN, deltas, len(x), learning_rate, weight_decay)

    return learning_curve


# initialize the network with 3 layers and delta=0 per neuron
def init_network(n_inputs, n_hidden, n_outputs):
    network = []
    hidden_layer = {'weights': np.random.normal(scale=1e-4, size=((n_inputs + 1) * n_hidden))
                                            .reshape((n_hidden, n_inputs+1))}
    network.append(hidden_layer)
    output_layer = {'weights': np.random.normal(scale=1e-4, size=((n_hidden + 1) * n_outputs))
                                            .reshape((n_outputs, n_hidden + 1))}
    network.append(output_layer)
    return network


# return the learning examples
def get_training_data():
    data = np.eye(8)
    np.random.shuffle(data)
    return data, data


def main():
    NN = init_network(8, 3, 8)
    x, y = get_training_data()

    GS = input('GridSearch (y/n): ')

    if GS.lower() == 'y':  # perform grid search
        errors = []

        rates = [0.1, 0.25, 0.5, 0.75, 0.9]
        iterations = [10, 100, 1000, 2000, 5000]
        decays = [0., 0.001, 0.01, 0.1]

        for i, rate in enumerate(rates):
            print('%d / %d rates' % (i+1, len(rates)))
            for j, decay in enumerate(decays):
                for k, iter in enumerate(iterations):
                    NN = init_network(8, 3, 8)
                    start_time = time.time()
                    sum_error = gradient_descent(NN, x, y, iter, rate, decay)[-1]

                    elapsed = time.time() - start_time
                    errors.append((rate, decay, iter, sum_error, elapsed))
                    print(errors[-1])
            print('-' * 20)

        best_score = (0, 0, 0, float('inf'), 0)
        best_time = (0, 0, 0, 0, float('inf'))
        for er in errors:
            if er[3] < best_score[3]:
                best_score = er
            if er[4] < best_time[4]:
                best_time = er

        print(best_score)
        print(best_time)

    else:  # do not perform grid search
        learning_curve = gradient_descent(NN, x, y, 2000, 0.5, 0.001)
        plt.plot(np.arange(len(learning_curve)), learning_curve)
        plt.show()

        print('score: %.5f' % learning_curve[-1])




if __name__ == '__main__':
    # reproducible results
    np.random.seed(42)

    main()
