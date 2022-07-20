import matplotlib.pyplot as plt
import numpy as np

# To ensure reproducibility
np.random.seed(42)

neural_network = [
    {"in": 784, "out": 16, "activation": "relu"},
    {"in": 16, "out": 10, "activation": "sigmoid"},
]


from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from joblib import Memory


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
    X_test = split_into_batches(min_max_scaler.fit_transform(np.array(X_test)), 1)

    # Turn the targets into Numpy arrays and flatten the array
    y_train = np.array(y_train).reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)

    # One-Hot encode the training data and split it into batches (same as with the training data)
    one_hot_encoder.fit(y_train)
    y_train = one_hot_encoder.transform(y_train).toarray()
    y_train = split_into_batches(np.array(y_train), batch_size)

    one_hot_encoder.fit(y_test)
    y_test = one_hot_encoder.transform(y_test).toarray()
    y_test = split_into_batches(np.array(y_test), 1)

    return X_train, y_train, X_test, y_test


def initialise_weights_and_biases(nn_architecture):
    parameters = {}
    for idx, layer in enumerate(nn_architecture):
        n_input = layer["in"]
        n_output = layer["out"]

        parameters[f"weights {idx}->{idx+1}"] = np.random.randn(n_input, n_output) * 0.1
        parameters[f"bias {idx+1}"] = (
            np.random.randn(
                n_output,
            )
            * 0.1
        )

    return parameters


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def relu(Z):
    return np.maximum(0, Z)


def main():
    neural_network = [
        {"in": 784, "out": 16, "activation": "relu"},
        {"in": 16, "out": 10, "activation": "sigmoid"},
    ]

    parameters = initialise_weights_and_biases(neural_network)


if __name__ == "__main__":
    main()
