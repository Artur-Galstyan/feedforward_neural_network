import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from joblib import Memory

# This is the same content as neural_network.py except everything was wrapped in a class.


def get_mnist(batch_size=64, random_seed=42):
    def split_into_batches(x, batch_size):
        n_batches = len(x) / batch_size
        x = np.array_split(x, n_batches)
        return np.array(x, dtype=object)

    # To cache the downloaded data
    memory = Memory("./mnist")
    fetch_openml_cached = memory.cache(fetch_openml)
    mnist = fetch_openml_cached("mnist_784")

    # Split the data into training and validation sets
    X_train, X_test, y_train, y_test = train_test_split(
        mnist.data, mnist.target, test_size=0.33, random_state=random_seed
    )

    # Normalizes the data
    min_max_scaler = MinMaxScaler()

    # One-Hot encodes the targets
    one_hot_encoder = OneHotEncoder()

    # Split the training data into batches
    X_train = split_into_batches(
        min_max_scaler.fit_transform(np.array(X_train)), batch_size
    )

    #
    X_test = split_into_batches(
        min_max_scaler.fit_transform(np.array(X_test)), batch_size
    )

    # Turn the targets into Numpy arrays and flatten the array
    y_train = np.array(y_train).reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)

    # One-Hot encode the training data and split it into batches (same as with the training data)
    one_hot_encoder.fit(y_train)
    y_train = one_hot_encoder.transform(y_train).toarray()
    y_train = split_into_batches(np.array(y_train), batch_size)

    one_hot_encoder.fit(y_test)
    y_test = one_hot_encoder.transform(y_test).toarray()
    y_test = split_into_batches(np.array(y_test), batch_size)

    return X_train, y_train, X_test, y_test


class NeuralNetwork:
    def __init__(self, nn_architecture):
        self.nn_architecture = nn_architecture
        self.parameters = self.initialise_weights_and_biases()

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def relu(self, Z):
        return np.maximum(0, Z)

    def sigmoid_backward(self, dA, Z):
        sig = self.sigmoid(Z)
        return dA * sig * (1 - sig)

    def relu_backward(self, dA, Z):
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ

    def initialise_weights_and_biases(self):
        parameters = {}
        for idx, layer in enumerate(self.nn_architecture):
            n_input = layer["in"]
            n_output = layer["out"]

            parameters[f"weights {idx}->{idx+1}"] = (
                np.random.randn(n_input, n_output) * 0.1
            )
            parameters[f"bias {idx+1}"] = (
                np.random.randn(
                    n_output,
                )
                * 0.1
            )

        return parameters

    def forward_single_layer(self, a_prev, w, b, activation_function):
        z = a_prev @ w + b
        a = activation_function(z)

        return a, z

    def forward(self, x):
        # If our network has 3 layers, our dictionary has only 2 entries.
        # Therefore we need to add 1 on top
        n_layers = len(self.nn_architecture) + 1
        # The memory is needed later for backpropagation
        memory = {}
        a = x

        # We have 3 layers, 0, 1 and 2 and want to skip 0
        # Therefore we start at 1
        for i in range(1, n_layers):
            a_prev = a
            activation_function = self.__getattribute__(
                self.nn_architecture[i - 1]["activation"]
            )
            w = self.parameters[f"weights {i-1}->{i}"]
            b = self.parameters[f"bias {i}"]

            a, z = self.forward_single_layer(a_prev, w, b, activation_function)

            memory[f"a_{i - 1}"] = a_prev
            memory[f"z_{i}"] = z

        return a, memory

    def backpropagation_single_layer(self, dA, w, z, a_prev, activation_function):
        m = a_prev.shape[0]
        backprop_activation = self.__getattribute__(f"{activation_function}_backward")

        delta = backprop_activation(dA, z)
        dW = (a_prev.T @ delta) / m
        dB = np.sum(delta, axis=1, keepdims=True) / m
        dA_prev = delta @ w.T

        return dW, dB, dA_prev

    def backward(self, target, prediction, memory):
        gradients = {}
        dA_prev = 2 * (prediction - target)
        # If our network has 3 layers, our dictionary has only 2 entries.
        # Therefore we need to add 1 on top
        n_layers = len(self.nn_architecture) + 1

        # Loop backwards
        for i in reversed(range(1, n_layers)):
            dA = dA_prev

            # Memory from the forward propagation step
            a_prev = memory[f"a_{i-1}"]
            z = memory[f"z_{i}"]

            w = self.parameters[f"weights {i-1}->{i}"]

            dW, dB, dA_prev = self.backpropagation_single_layer(
                dA, w, z, a_prev, self.nn_architecture[i - 1]["activation"]
            )

            gradients[f"dW_{i-1}->{i}"] = dW
            gradients[f"dB_{i}"] = dB

        return gradients

    def update(self, gradients, learning_rate):
        n_layers = len(self.nn_architecture) + 1
        for i in range(1, n_layers):
            self.parameters[f"weights {i-1}->{i}"] -= (
                learning_rate * gradients[f"dW_{i-1}->{i}"]
            )
            self.parameters[f"bias {i}"] -= learning_rate * gradients[f"dB_{i}"].mean()

    def get_current_accuracy(self, X_test, y_test):
        correct = 0
        total_counter = 0
        for x, y in zip(X_test, y_test):
            a, _ = self.forward(x)
            pred = np.argmax(a, axis=1, keepdims=True)
            y = np.argmax(y, axis=1, keepdims=True)
            correct += (pred == y).sum()
            total_counter += len(x)
        accuracy = correct / total_counter
        return accuracy

    def train(self, x, y, learning_rate=0.1):
        a, memory = self.forward(x)
        grads = self.backward(y, a, memory)
        self.update(grads, learning_rate)


def save_object(obj, filename):
    with open(filename, "wb") as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def main():
    neural_network = NeuralNetwork(
        [
            {"in": 784, "out": 128, "activation": "relu"},
            {"in": 128, "out": 32, "activation": "relu"},
            {"in": 32, "out": 16, "activation": "relu"},
            {"in": 16, "out": 10, "activation": "sigmoid"},
        ]
    )
    X_train, y_train, X_test, y_test = get_mnist(batch_size=128)

    n_epochs = 50
    for epoch in range(n_epochs):
        for x, y in zip(X_train, y_train):
            neural_network.train(x, y)
        accuracy = neural_network.get_current_accuracy(X_test, y_test)
        print(f"Epoch {epoch} Accuracy = {np.round(accuracy * 100, 4) }%")

    save_object(
        {
            "architecture": neural_network.nn_architecture,
            "parameters": neural_network.parameters,
        },
        "nn_pickle",
    )


if __name__ == "__main__":
    main()
