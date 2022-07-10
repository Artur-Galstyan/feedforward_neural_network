import numpy as np
import torch_helpers.TorchDataset as td
import matplotlib.pyplot as plt
from tqdm import tqdm


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x):
    return np.maximum(0, x)


def deriv_relu(x):
    return x > 0


n_samples = 200
n_epochs = 5000
learning_rate = 0.01
batch_size = 1

n_input = 2
n_hidden = 16
n_output = 2

simple_x = [np.random.uniform(-2, 2, size=(n_input)).round(2) for _ in range(n_samples)]
simple_y = [i * 2 + np.random.uniform(-0.05, 0.05) for i in simple_x]


training_set, test_set = td.TorchDataset(
    simple_x, simple_y
).get_train_and_validation_data(batch_size=batch_size, validation_ratio=0.2)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


class NeuralNetwork:
    def __init__(self):
        self.w0 = np.random.randn(n_input, n_hidden)
        self.b0 = np.random.randn(n_hidden)
        self.w1 = np.random.randn(n_hidden, n_output)
        self.b1 = np.random.randn(n_output)

    def forward(self, x):
        self.z0 = x @ self.w0 + self.b0
        self.a0 = sigmoid(self.z0)

        self.z1 = self.a0 @ self.w1 + self.b1
        return self.z1

    def backward(self, x, y):
        delta_1 = np.multiply(2 * (self.z1 - y), 1)
        self.delta_w1 = self.a0.T @ delta_1
        self.delta_b1 = delta_1.sum(axis=0).mean()

        delta_0 = np.multiply(delta_1 @ self.w1.T, sigmoid_derivative(self.z0))
        self.delta_w0 = x.T @ delta_0
        self.delta_b0 = delta_0.sum(axis=0).mean()

    def update(self, learning_rate):
        self.w0 -= learning_rate * self.delta_w0
        self.w1 -= learning_rate * self.delta_w1

        self.b0 -= learning_rate * self.delta_b0
        self.b1 -= learning_rate * self.delta_b1


nn = NeuralNetwork()

costs = []
for epoch in tqdm(range(n_epochs)):
    for x, y in training_set:
        x = x.numpy()
        y = y.numpy()
        prediction = nn.forward(x)

        costs.append(((y - prediction) ** 2).mean())

        nn.backward(x, y)
        nn.update(learning_rate)

for x, y in test_set:
    x = x.numpy()
    y = y.numpy()

    pred = nn.forward(x)

    print(f"x = {x}; Expected {np.round(y, 2)}; Got {np.round(pred, 2)}")

plt.plot(costs)
plt.show()
