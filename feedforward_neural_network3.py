import numpy as np
import torch_helpers.TorchDataset as td
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

np.random.seed(42)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x):
    return np.maximum(0, x)


def deriv_relu(x):
    return x > 0


def softmax(x):
    """Compute the softmax of vector x."""
    exps = np.exp(x - x.max())
    return exps / np.sum(exps)


def deriv_softmax(x):
    s = softmax(x)
    a = np.eye(s.shape[-1])
    temp1 = np.zeros((s.shape[0], s.shape[1], s.shape[1]), dtype=np.float32)
    temp2 = np.zeros((s.shape[0], s.shape[1], s.shape[1]), dtype=np.float32)
    temp1 = np.einsum("ij,jk->ijk", s, a)
    temp2 = np.einsum("ij,ik->ijk", s, s)
    return (temp1 - temp2).reshape(s.shape[0], s.shape[1])


n_epochs = 100
learning_rate = 0.1
batch_size = 256

n_input = 784
n_hidden_1 = 64
n_hidden_2 = 16
n_output = 10

mnist_trainset = datasets.MNIST(
    root="./data", train=True, download=True, transform=None
)
mnist_testset = datasets.MNIST(
    root="./data", train=False, download=True, transform=None
)

training_set_x = mnist_trainset.data.reshape(len(mnist_trainset), n_input) / 255
training_set_y = np.array(
    [[0 if i != y else 1 for i in range(10)] for y in mnist_trainset.targets]
)


torch_dataset = td.TorchDataset(training_set_x, training_set_y)
training_data, _ = torch_dataset.get_train_and_validation_data(
    batch_size=batch_size, validation_ratio=0
)

testing_set_x = mnist_testset.data.reshape(len(mnist_testset), n_input) / 255
testing_set_y = np.array(
    [[0 if i != y else 1 for i in range(10)] for y in mnist_testset.targets]
)


torch_dataset = td.TorchDataset(testing_set_x, testing_set_y)
testing_data, _ = torch_dataset.get_train_and_validation_data(
    batch_size=1, validation_ratio=0
)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


class NeuralNetwork:
    def __init__(self):
        self.w0 = np.random.randn(n_input, n_hidden_1)
        self.b0 = np.random.randn(n_hidden_1)
        self.w1 = np.random.randn(n_hidden_1, n_hidden_2)
        self.b1 = np.random.randn(n_hidden_2)
        self.w2 = np.random.randn(n_hidden_2, n_output)
        self.b2 = np.zeros((n_output))

    def forward(self, x):
        self.z0 = x @ self.w0 + self.b0
        self.a0 = relu(self.z0)

        self.z1 = self.a0 @ self.w1 + self.b1
        self.a1 = relu(self.z1)

        self.z2 = self.a1 @ self.w2 + self.b2
        self.a2 = self.z2
        return self.a2

    def backward(self, x, y):
        delta_2 = np.multiply(2 * (self.a2 - y), 1)
        self.delta_w2 = (self.a1.T @ delta_2) / len(x)
        self.delta_b2 = delta_2.sum(axis=0).mean()

        delta_1 = np.multiply(delta_2 @ self.w2.T, deriv_relu(self.z1))
        self.delta_w1 = (self.a0.T @ delta_1) / len(x)
        self.delta_b1 = delta_1.sum(axis=0).mean()

        delta_0 = np.multiply(delta_1 @ self.w1.T, deriv_relu(self.z0))
        self.delta_w0 = (x.T @ delta_0) / len(x)
        self.delta_b0 = delta_0.sum(axis=0).mean()

    def update(self, learning_rate):
        self.w0 -= learning_rate * self.delta_w0
        self.w1 -= learning_rate * self.delta_w1
        self.w2 -= learning_rate * self.delta_w2

        self.b0 -= learning_rate * self.delta_b0
        self.b1 -= learning_rate * self.delta_b1
        # self.b2 -= learning_rate * self.delta_b2


nn = NeuralNetwork()

costs = []
for epoch in tqdm(range(n_epochs)):
    avg_cost = 0
    for x, y in training_data:
        x = x.numpy()
        y = y.numpy().reshape(y.shape[0], y.shape[-1])
        prediction = nn.forward(x)

        avg_cost += ((y - prediction) ** 2).mean()

        nn.backward(x, y)
        nn.update(learning_rate)
    costs.append(avg_cost / len(training_data))

correct = 0
for x, y in testing_data:
    x = x.numpy()
    y = y.numpy().reshape(y.shape[0], y.shape[-1])
    pred = nn.forward(x)

    if np.argmax(y) == np.argmax(pred):
        correct += 1

    print(f"Expected {np.argmax(y)}; Got {np.argmax(pred)}")

print(f"Accuracy {correct / len(mnist_testset)}")

plt.plot(costs)
plt.show()
