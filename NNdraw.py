from matplotlib import pyplot
from math import cos, sin, atan

MIN = float('-inf')
MAX = float('inf')

class Neuron():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self, neuron_radius):
        circle = pyplot.Circle((self.x, self.y), radius=neuron_radius, fill=False)
        pyplot.gca().add_patch(circle)


class Layer():
    def __init__(self, network, number_of_neurons, number_of_neurons_in_widest_layer, weights):
        self.vertical_distance_between_layers = 6
        self.horizontal_distance_between_neurons = 2
        self.neuron_radius = 0.5
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()
        self.neurons = self.__intialise_neurons(number_of_neurons)
        self.weights = weights
        if self.weights:
            self.scalar = self.__init_percent(self.weights)
        else:
            self.scalar = None

    def __intialise_neurons(self, number_of_neurons):
        neurons = []
        x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for iteration in range(number_of_neurons):
            neuron = Neuron(x, self.y)
            neurons.append(neuron)
            x += self.horizontal_distance_between_neurons
        return neurons

    def __init_percent(self, weights):
        min_w = float('inf')
        max_w = float('-inf')

        for c1, neuron in enumerate(weights):
            for c2, connection in enumerate(neuron):
                w = weights[c1]['weights'][0][c2][0]
                if w > max_w:
                    max_w = w
                if w < min_w:
                    min_w = w

        max_w = 1 / (max_w + abs(min_w))

        return abs(min_w), max_w


    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        return self.horizontal_distance_between_neurons * (self.number_of_neurons_in_widest_layer - number_of_neurons) / 2

    def __calculate_layer_y_position(self):
        if self.previous_layer:
            return self.previous_layer.y + self.vertical_distance_between_layers
        else:
            return 0

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def __line_between_two_neurons(self, neuron1, neuron2, color):
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = self.neuron_radius * sin(angle)
        y_adjustment = self.neuron_radius * cos(angle)
        line = pyplot.Line2D((neuron1.x - x_adjustment, neuron2.x + x_adjustment),
                             (neuron1.y - y_adjustment, neuron2.y + y_adjustment),
                             color=color)
        pyplot.gca().add_line(line)

    def draw(self, layerType=0):
        for count, neuron in enumerate(self.neurons):
            neuron.draw( self.neuron_radius )
            if self.previous_layer:
                for count2, previous_layer_neuron in enumerate(self.previous_layer.neurons):
                    col = (0, 0, 1)
                    if self.weights:
                        percent = self.percentage(self.weights[count]['weights'][0][count2][0])
                        if (percent < 0) | (percent > 1):
                            percent = round(percent, 0)
                        if percent < 0.5:
                            col = (0, percent+0.25, 0)
                        elif percent > 0.5:
                            col = (percent-0.25, 0, 0)
                        else:
                            col = (0, 0, percent/0.1)
                    self.__line_between_two_neurons(neuron, previous_layer_neuron, col)

        # write Text
        x_text = self.number_of_neurons_in_widest_layer * self.horizontal_distance_between_neurons
        if layerType == 0:
            pyplot.text(x_text, self.y, 'Input Layer', fontsize = 12)
        elif layerType == -1:
            pyplot.text(x_text, self.y, 'Output Layer', fontsize = 12)
        else:
            pyplot.text(x_text, self.y, 'Hidden Layer '+str(layerType), fontsize = 12)

    def percentage(self, weight):
        w = weight + self.scalar[0]
        w *= self.scalar[1]
        return round(w, 1)


class NeuralNetwork():
    def __init__(self, number_of_neurons_in_widest_layer):
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.layers = []
        self.layertype = 0

    def add_layer(self, number_of_neurons, weights):
        layer = Layer(self, number_of_neurons, self.number_of_neurons_in_widest_layer, weights)
        self.layers.append(layer)

    def draw(self):
        pyplot.figure()
        for i in range( len(self.layers) ):
            layer = self.layers[i]
            if i == len(self.layers)-1:
                i = -1
            layer.draw( i )
        pyplot.axis('scaled')
        pyplot.axis('off')
        pyplot.title( 'Neural Network architecture', fontsize=15 )
        pyplot.show()


class DrawNN:
    def __init__(self, neural_network, network):
        self.neural_network = neural_network
        self.my_net = network

    def draw(self):
        widest_layer = max(self.neural_network)
        network = NeuralNetwork(widest_layer)
        for l, n in enumerate(self.neural_network):
            if l != 0:
                weights = self.my_net[l-1]
            else:
                weights = None
            network.add_layer(n, weights)
        network.draw()
