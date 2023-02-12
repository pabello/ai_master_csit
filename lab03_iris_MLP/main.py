import pandas as pd
import numpy as np
from math import exp, log
from copy import deepcopy
from datetime import datetime
import utils


class MultilayerPerceptron:
    # Activation functions
    def __relu(inputs: np.ndarray):
        return np.matrix([max(0, x) for x in inputs])

    def __softmax (inputs:np.ndarray):
        denominator = sum(np.array(list(map(exp, inputs))))
        return np.array([exp(x) / denominator for x in inputs])

    # Derivatives of activation functions
    def __relu_derivative(inputs:np.ndarray):
        return np.asmatrix(inputs > 0)
    
    def __softmax_derivative(softmax_outputs:np.ndarray):
        subtractor = [[softmax_outputs[i] if i==j else 0 for i in range(len(softmax_outputs))] for j in range(len(softmax_outputs))]
        predictions = np.matrix(softmax_outputs)
        derivative_matrix = -predictions.T @ predictions - subtractor
        return derivative_matrix
        

    # Error functions
    def __MSE(self, outputs:np.ndarray, reference:np.ndarray):
        return ((reference - outputs)**2).mean()
    
    def __MSE_derivative(self, outputs:np.ndarray, reference:np.ndarray):
        return np.asmatrix(outputs - reference) * 2

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

        self.filename = f"{inputs_number}."
        self.weights = list()
        self.biases = list()
        self.neuron_values = list()  # neuron values pre activation funciton
        self.layers_outputs = list()  # neuron values after activation function
        self.learning_factor = learning_factor
        self.activation_function = MultilayerPerceptron.__relu
        self.activation_function_derivative = MultilayerPerceptron.__relu_derivative
        self.output_layer_activation_function = MultilayerPerceptron.__softmax
        self.output_layer_activation_function_derivative = MultilayerPerceptron.__softmax_derivative
        self.error_function = MultilayerPerceptron.__MSE

        # Creating a new model
        previous_layer_neurons = inputs_number
        if not weights:
            layer_sizes = list(hidden_layers_sizes) + [outputs_number]
            for size in layer_sizes:
                # self.weights.append(np.random.uniform(size=(size, previous_layer_neurons), low=-1, high=1))
                self.weights.append(np.random.normal(size=(size, previous_layer_neurons), loc=0.0, scale=0.5))
                # self.weights.append(np.random.randn(size, previous_layer_neurons))
                self.biases.append(np.matrix(np.zeros(size)).T)
                previous_layer_neurons = size
                self.filename += str(size) + '.'
        # Loading a previously trained model
        else:
            self.weights = weights
            self.biases = biases
    
    def feed_forward(self, inputs:np.ndarray):
        layer_input = inputs
        # print(layer_input)
        self.neuron_values.append(inputs)
        self.layers_outputs.append(inputs)
        
        for weights, biases in zip(self.weights[:-1], self.biases[:-1]):
            # print(layer_input)
            
            # print("weights")
            # print(weights)
            # print(type(weights))
            # print(weights.shape)
            # print()
            # print("layer_input")
            # print(layer_input.T)
            # print(type(layer_input.T))
            # print(layer_input.T.shape)
            # print()
            # print("(weights @ layer_input.T)")
            # print((weights @ layer_input.T))
            # print(type((weights @ layer_input.T)))
            # print((weights @ layer_input.T).shape)
            # print()
            # print("biases")
            # print(biases.T)
            # print(type(biases.T))
            # print(biases.T.shape)
            # print()
            self.neuron_values.append((weights @ layer_input.T) + biases)
            # print()
            # print("self.neuron_values[-1]")
            # print(self.neuron_values[-1])
            # print(type(self.neuron_values[-1]))
            # print(self.neuron_values[-1].shape)
            # print("--------------------")
            # print()
            values_array = np.asarray(self.neuron_values[-1].flatten())[0]
            self.layers_outputs.append(self.activation_function(values_array))
            layer_input = self.layers_outputs[-1]
            # print(layer_input)
        # print(layer_input)
        
        network_output = self.output_layer_activation_function(self.weights[-1] @ layer_input.T + self.biases[-1])
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
    
    def backpropagate(self, prediction:np.ndarray, reference:np.ndarray):
        cost_value = ((prediction - reference)**2).sum()
        cost_derivative = 2 * (prediction - reference)
        
        bias_changes = []
        weight_changes = []
        
        mse = self.__MSE(prediction, reference)
        mse_derivative = self.__MSE_derivative(prediction, reference)
        
        output_layer_derivative = self.output_layer_activation_function_derivative(prediction)
        neuron_changes = output_layer_derivative @ mse_derivative.T
        
        for i in reversed(range(len(self.weights))):
            # print(f"i: {i}")
            weights = self.weights[i]
            layer_output = np.asmatrix(self.layers_outputs[i])
            neurons_activation_derivative = self.activation_function_derivative(self.neuron_values[i])
            
            bias_changes.append(np.array(neuron_changes))
            weight_changes.append(neuron_changes @ layer_output)
            neuron_changes = np.multiply((weights.T @ neuron_changes), neurons_activation_derivative)
        
        return list(reversed(weight_changes)), list(reversed(bias_changes))

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

    def update_network(self, weight_changes:np.matrix, bias_changes:np.matrix):
        for i in range(len(self.weights)):
            self.weights[i] += weight_changes[i]
            self.biases[i] += bias_changes[i]

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

    
    def train(self, df:pd.DataFrame, epochs=100, error_change=0.001):
        # epochs = 10
        for epoch in range(epochs):
            df = df.sample(frac=1).reset_index(drop=True)  # shuffling the dataset
            error = 0
            for i in range(len(df)):
                reference = np.array(df.iloc[i][4])
                prediction = self.feed_forward(np.matrix(df.iloc[i][:4]))
                weight_changes, bias_changes = self.backpropagate(prediction, reference)
                # print(prediction)
                # print(reference)
                # print(prediction - reference)
                # print()
                error += sum((prediction - reference) ** 2)
                
                for i in range(len(self.weights)):
                    # These two cannot utilize the short notation -= due to type incompatibility
                    self.weights[i] = self.weights[i] + weight_changes[i] * self.learning_factor
                    self.biases[i] = self.biases[i] + bias_changes[i] * self.learning_factor
            print(f"Epoch: {epoch} | Mean error: {error / len(df)}")
                
        # with open(f"MLP_self.filename_EPOCHS={epochs}_STAMP={int(datetime.now().timestamp()*1_000_000)}", "wb") as file:
        #     results = np.array([self.weights, self.biases])
        #     np.save(file, results)

    # def test_model(self):
    #     df = pd.read_excel('data/pozyxAPI_only_localization_dane_testowe_i_dystrybuanta.xlsx', usecols=[4,5,6,7]).dropna()
    #     self.data = pd.DataFrame.to_numpy(df.sample(frac=1))

    #     self.test_outputs = []


    # def get_data(self, path):
    #     df = load_data(path)
    #     self.data = pd.DataFrame.to_numpy(df)



if __name__ == "__main__":
    np.random.seed(6)
    
    # network = MultilayerPerceptron(4, 3, (5, 6))    
    # outputs = network.feed_forward(np.array([3.14, 2.17, 1.04, 1.43]))
    
    # # print("Outputs:")
    # # print(outputs)
    # # print()
    # # print(network.neuron_values)

    # print()
    # example = np.array([0.69, .21, .10])
    # network.backpropagate(outputs, np.array([0, 1, 0]))
    
    training_set, testing_set = utils.load_iris_dataset("iris.data", 0.8)
    network = MultilayerPerceptron(inputs_number=4,
                                    outputs_number=3,
                                    hidden_layers_sizes=(5, 6),
                                    learning_factor=0.1)
    network.train(training_set, epochs=100)
    