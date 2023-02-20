import pandas as pd
import numpy as np
from math import exp, log, isnan
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
                self.weights.append(np.asmatrix(np.random.normal(size=(size, previous_layer_neurons), loc=0.0, scale=0.5)))
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
        self.neuron_values.append(inputs)
        self.layers_outputs.append(inputs)
        
        for weights, biases in zip(self.weights[:-1], self.biases[:-1]):
            self.neuron_values.append((weights @ layer_input.T) + biases)
            values_array = np.asarray(self.neuron_values[-1].flatten())[0]
            self.layers_outputs.append(self.activation_function(values_array))
            layer_input = self.layers_outputs[-1]
        
        network_output = self.output_layer_activation_function(self.weights[-1] @ layer_input.T + self.biases[-1])
        return network_output
    
    
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
    
    
    def update_network(self, weight_changes:np.matrix, bias_changes:np.matrix):
        for i in range(len(self.weights)):
            self.weights[i] += weight_changes[i]
            self.biases[i] += bias_changes[i]
    
    
    def train(self, df:pd.DataFrame, epochs=100, error_change=0.001, testing_data=None):
        for epoch in range(epochs):
            df = df.sample(frac=1).reset_index(drop=True)  # shuffling the dataset
            error = 0
            weight_changes_sum = [self.weights[i] * 0 for i in range(len(self.weights))]
            bias_changes_sum = [self.biases[i] * 0 for i in range(len(self.biases))]
            sample_counter = 0
            for i in range(len(df)):
                reference = np.array(df.iloc[i][4])
                prediction = self.feed_forward(np.matrix(df.iloc[i][:4]))
                
                weight_changes, bias_changes = self.backpropagate(prediction, reference)
                error += sum((prediction - reference) ** 2)
                
                for layer in range(len(self.weights)):
                    weight_changes_sum[layer] = weight_changes_sum[layer] + weight_changes[layer]
                    bias_changes_sum[layer] = bias_changes_sum[layer] + bias_changes[layer]
                sample_counter += 1
                if not i % 32 or i == len(df) - 1:
                    for layer in range(len(self.weights)):
                        self.weights[layer] = self.weights[layer] + (weight_changes_sum[layer] / sample_counter) * self.learning_factor
                        self.biases[layer] = self.biases[layer] + (bias_changes_sum[layer] / sample_counter) * self.learning_factor
                        weight_changes_sum[layer] *= 0
                        bias_changes_sum[layer] *= 0
                    sample_counter = 0
                
                # # Weights and biases as list of matrixes
                # for i in range(len(self.weights)):
                #     # These two cannot utilize the short notation -= due to type incompatibility
                #     self.weights[i] = self.weights[i] + weight_changes[i] * self.learning_factor
                #     self.biases[i] = self.biases[i] + bias_changes[i] * self.learning_factor
            epoch_summary = f"Epoch: {epoch+1} | Mean error: {error / len(df):.6f}"
            if testing_data is not None:
                accuracy = network.test(testing_data)
                epoch_summary += f" | Model accuracy: {accuracy:.4f}"
            print(epoch_summary)
            if isnan(error):
                raise ValueError("Weights gone crazy, mean error not fitting in float scale. Shutting down.")
    
    
    def test(self, df:pd.DataFrame) -> float:
        """Method for testing the accuracy of the model.
        
        Args:
            df (pd.DataFrame): testing dataset
        
        Returns:
            float: accuracy of the model (correct predictions / total predictions)
        """
        df = df.sample(frac=1).reset_index(drop=True)  # shuffling the dataset
        
        correct_predictions = 0
        
        for i in range(len(df)):
            reference = np.array(df.iloc[i][4])
            prediction = self.feed_forward(np.matrix(df.iloc[i][:4]))
            
            if np.argmax(prediction) == np.argmax(reference):
                correct_predictions += 1
        return correct_predictions / len(df)


if __name__ == "__main__":
    # np.random.seed(6)
    
    training_set, testing_set = utils.load_iris_dataset("iris.data", 0.8)
    network = MultilayerPerceptron(inputs_number=4,
                                    outputs_number=3,
                                    hidden_layers_sizes=(5, 6),
                                    learning_factor=0.001)
    network.train(training_set, epochs=100, testing_data=testing_set)
    