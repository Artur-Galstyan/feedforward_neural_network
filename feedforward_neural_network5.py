import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from joblib import Memory


def split_into_batches(x, batch_size):
    n_batches = len(x) / batch_size
    x = np.array_split(x, n_batches)
    return np.array(x)


def init_params(nn_architecture):
    param_values = {}

    for idx, layer in enumerate(nn_architecture):
        l_input = layer["input_dim"]
        l_output = layer["output_dim"]

        param_values[f"w_{str(idx)}"] = np.random.randn(l_input, l_output) * 0.1
        param_values[f"b_{str(idx)}"] = (
            np.random.randn(
                l_output,
            )
            * 0.1
        )

    return param_values


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def relu(Z):
    return np.maximum(0, Z)


def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)


def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


def single_layer_forward_propagation(a_prev, w, b, activation_function):
    z = a_prev @ w + b
    if activation_function == "sigmoid":
        activation = sigmoid
    elif activation_function == "relu":
        activation = relu

    a = activation(z)

    return a, z


def full_forward_propagation(x, param_values, nn_architecture):
    x = x.reshape(-1, 1).T
    memory = {}
    a = x

    for idx, layer in enumerate(nn_architecture):
        a_prev = a
        activation_function = layer["activation"]

        w = param_values[f"w_{idx}"]
        b = param_values[f"b_{idx}"]

        a, z = single_layer_forward_propagation(a_prev, w, b, activation_function)

        memory[f"a_{idx - 1}"] = a_prev
        memory[f"z_{idx}"] = z

    return a, memory


def single_layer_backpropagation(dA, w, z, a_prev, activation_function):
    m = a_prev.shape[0]
    if activation_function == "relu":
        backprop_activation = relu_backward
    elif activation_function == "sigmoid":
        backprop_activation = sigmoid_backward

    delta = backprop_activation(dA, z)
    dW = (a_prev.T @ delta) / m
    dB = np.sum(delta, axis=1, keepdims=True) / m
    dA_prev = delta @ w.T

    return dW, dB, dA_prev


def full_backpropagation(target, prediction, memory, param_values, nn_architecture):
    gradients = {}
    m = prediction.shape[0]

    dA_prev = 2 * (prediction - target)

    for idx, layer in reversed(list(enumerate(nn_architecture))):
        idx -= 1
        activation_function = layer["activation"]

        dA = dA_prev

        a_prev = memory[f"a_{idx}"]
        z = memory[f"z_{idx + 1}"]
        w = param_values[f"w_{idx + 1}"]

        dW, dB, dA_prev = single_layer_backpropagation(
            dA, w, z, a_prev, activation_function
        )

        gradients[f"dW_{idx}"] = dW
        gradients[f"dB_{idx}"] = dB

    return gradients


def update(param_values, gradients, nn_architecture, learning_rate):
    for idx, layer in enumerate(nn_architecture):
        param_values[f"w_{idx}"] -= learning_rate * gradients[f"dW_{idx-1}"]
        param_values[f"b_{idx}"] -= learning_rate * gradients[f"dB_{idx-1}"].mean()
    return param_values


def main():
    memory = Memory("./mnist")

    np.random.seed(42)
    batch_size = 1

    nn_architecture = [
        {"input_dim": 784, "output_dim": 64, "activation": "relu"},
        {"input_dim": 64, "output_dim": 32, "activation": "relu"},
        {"input_dim": 32, "output_dim": 16, "activation": "relu"},
        {"input_dim": 16, "output_dim": 10, "activation": "sigmoid"},
    ]

    param_values = init_params(nn_architecture)

    fetch_openml_cached = memory.cache(fetch_openml)
    mnist = fetch_openml_cached("mnist_784")

    X_train, X_test, y_train, y_test = train_test_split(
        mnist.data, mnist.target, test_size=0.33, random_state=42
    )

    minmax_scaler = MinMaxScaler()
    ohe = OneHotEncoder()

    X_train = split_into_batches(
        minmax_scaler.fit_transform(np.array(X_train)), batch_size
    )
    X_test = split_into_batches(minmax_scaler.fit_transform(np.array(X_test)), 1)

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    ohe.fit(y_train)
    y_train = ohe.transform(y_train).toarray()
    y_train = split_into_batches(np.array(y_train), batch_size)

    # Fit and transform testing data
    ohe.fit(y_test)
    y_test = ohe.transform(y_test).toarray()
    y_test = split_into_batches(np.array(y_test), 1)

    n_epochs = 200
    learning_rate = 0.1
    for epoch in tqdm(range(n_epochs)):

        for x, y in zip(X_train, y_train):
            a, memory = full_forward_propagation(x, param_values, nn_architecture)
            gradients = full_backpropagation(y, a, memory, param_values, nn_architecture)

            param_values = update(param_values, gradients, nn_architecture, learning_rate)
        


if __name__ == "__main__":
    main()
