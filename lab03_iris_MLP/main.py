import numpy as np
from math import exp, log
from copy import copy


class MultilayerPerceptron:
    # Activation functions
    def __relu(inputs: np.ndarray):
        return np.array([max(0, x) for x in inputs])

    def __softmax (inputs:np.ndarray):
        denominator = sum(np.array(list(map(exp, inputs))))
        return np.array([exp(x) / denominator for x in inputs])

    # Derivatives of activation functions
    def __relu_derivative(inputs:np.ndarray):
        return np.array([int(x) for x in (inputs > 0).tolist()])
    
    def __softmax_derivative(softmax_outputs:np.ndarray):
        subtractor = [[softmax_outputs[i] if i==j else 0 for i in range(len(softmax_outputs))] for j in range(len(softmax_outputs))]
        predictions = np.matrix(softmax_outputs)
        return -predictions.T @ predictions - subtractor
        

    # Error functions
    def __MSE(inputs:np.ndarray):
        return (inputs**2).mean()

    def __softmax_cross_entropy(softmax_values:np.ndarray, class_values:np.ndarray) -> np.ndarray:
        return -1 * class_values * np.array(list(map(log, softmax_values)))
    def __softmax_cross_entropy_derivative(softmax_values:np.ndarray, raw_values:np.ndarray):
        return -1 / softmax_values * (raw_values * (1 - raw_values))
    
    # def __softmax_derivative(softmax_values:list[float]):
    #     output = []
    #     output.append(softmax_values[0] * (1-softmax_values[0]))
    #     for value in softmax_values[1:]:
    #         output.append(-1 * softmax_values[0])
    #     return [softmax_values[0] * softmax_values[]]


    def __init__(self, inputs_number:int, outputs_number:int, hidden_layers_sizes:tuple[int], learning_factor:float=.1, weights=None, biases=None):
        """
        Instantiates an MLP - a Multilayer Perceptron.
        @requires inputs_number - number of input data pieces
        @requires outputs_number - number of output neurons
        @requires args - set of layer sizes one by one
        """
        '6.5.2_factor=0.1_take-0.npy'

        self.filename = ''
        self.weights = list()
        self.biases = list()
        self.neuron_values = list()
        self.layers_outputs = list()
        self.learning_factor = learning_factor
        self.activation_function = MultilayerPerceptron.__relu
        self.activation_function_derivative = MultilayerPerceptron.__relu_derivative
        self.output_layer_activation_function = MultilayerPerceptron.__softmax
        self.error_function = MultilayerPerceptron.__MSE
        previous_layer_neurons = inputs_number

        # Creating a new model
        if not weights:
            for size in hidden_layers_sizes:
                self.weights.append(np.random.uniform(size=(size, previous_layer_neurons), low=-1, high=1))
                self.biases.append(np.ones(size))
                previous_layer_neurons = size
                self.filename += str(size) + '.'
            self.filename += str(outputs_number) + f'_factor={learning_factor}_'
            self.weights.append(np.random.uniform(size=(outputs_number, previous_layer_neurons), low=-1, high=1))
            self.biases.append(np.ones(outputs_number))
        # Loading a previously trained model
        else:
            self.weights = weights
            self.biases = biases
    
    def feed_forward(self, inputs:np.ndarray):
        layer_input = inputs
        print(layer_input)
        
        for weights, biases in zip(self.weights[:-1], self.biases[:-1]):
            self.neuron_values.append((weights @ layer_input) + biases)
            self.layers_outputs.append(self.activation_function(self.neuron_values[-1]))
            layer_input = self.layers_outputs[-1]
            print(layer_input)
        
        network_output = self.output_layer_activation_function(self.weights[-1] @ layer_input)
        return network_output

    # def feed_forward(self, inputs):
    #     layer_input = inputs

    #     for weights, biases in zip(self.weights[:-1], self.biases[:-1]):
    #         sigma = weights @ layer_input + biases
    #         output = np.array( list( map(self.activation_function, sigma)))
    #         self.pre_squashing.append(sigma)
    #         self.outputs.append(output)
    #         layer_input = output

    #     output = self.weights[-1] @ layer_input + self.biases[-1]
    #     return output
    
    def backpropagate(self, input:np.ndarray, output:np.ndarray, reference:np.ndarray):
        cost_value = ((output - reference)**2).sum()
        
        output_layer_derivative = 2 * (output - reference)
        
        for neuron_values, layer_output, i in \
            zip(self.neuron_values[::-1], self.layers_outputs[::-1], reversed(range(len(self.layers_outputs)))):
                neuron_value_derivative = self.__relu_derivative(neuron_values)
                weights_derivative = layer_output
                

    # def backpropagate(self, input, output, reference):
    #     self.layer_errors.append((output - reference) * self.activation_function_derivative(output))
    #     # self.layer_errors.append(output - reference)

    #     # calculating node errors
    #     for sigma, activation, i in zip(self.pre_squashing[::-1], self.outputs[::-1], reversed(range(len(self.outputs)+1))):
    #         if self.activation_function == MultilayerPerceptron.__sigmoid:
    #             derivative_values = activation * (1 - activation)
    #         elif self.activation_function == MultilayerPerceptron.__tanh:
    #             derivative_values = 1 - activation**2
    #         else:
    #             derivative_values = self.activation_function_derivative(sigma)

    #         self.layer_errors.append(self.weights[i].T @ self.layer_errors[-1] * derivative_values)
    #     self.layer_errors.reverse()

    #     # calculating weights changes
    #     for i in range(len(self.weights)):
    #         if i == 0:
    #             input_array = input
    #         else:
    #             input_array = self.outputs[i-1]

    #         self.weights[i] += np.matrix(self.layer_errors[i]).T @ np.matrix(input_array * self.learning_factor)
    #         self.biases[i] += self.layer_errors[i] * self.learning_factor


    # def train(self, epochs, number):
    #     self.data = self.data / 1000;

    #     for epoch in range(epochs):
    #         np.random.shuffle(self.data)

    #         for record in enumerate(self.data):
    #             self.pre_squashing = []
    #             self.layer_errors = []
    #             self.outputs = []
    #             measurement = np.array([ record[1][0], record[1][1] ])
    #             reference = np.array([ record[1][2], record[1][3] ])

    #             output = self.feed_forward(measurement)
    #             self.backpropagate(measurement, output, reference)

    #     with open(self.filename+'take-{}'.format(number), 'wb') as file:
    #         results = np.array([self.weights, self.biases])
    #         np.save(file, results)

    # def test_model(self):
    #     df = pd.read_excel('data/pozyxAPI_only_localization_dane_testowe_i_dystrybuanta.xlsx', usecols=[4,5,6,7]).dropna()
    #     self.data = pd.DataFrame.to_numpy(df.sample(frac=1))

    #     self.test_outputs = []


    # def get_data(self, path):
    #     df = load_data(path)
    #     self.data = pd.DataFrame.to_numpy(df)



if __name__ == "__main__":
    np.random.seed(6)
    
    network = MultilayerPerceptron(4, 3, (5, 6))
    # print(network.weights)
    
    ff = network.feed_forward(np.array([3.14, 2.17, 1.04, 1.43]))
    error = network.error_function(ff)
    
    # print(ff)
    # print()
    # print(error)

    print()
    print(MultilayerPerceptron.softmax_derivative(np.array([0.69, .21, .10]), np.array([1, 0, 0])))