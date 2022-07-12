import numpy as np
from dataset import SupervisedData


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))


n_input = 784
n_hidden_1 = 16
n_output = 10

w_input_hidden = np.random.randn(n_input, n_hidden_1)
w_hidden_output = np.random.randn(n_hidden_1, n_output)

b_hidden = np.random.randn(n_hidden_1)
b_output = np.random.randn(n_output)

data = [np.random.randint(-3, 3) for _ in range(60000)]
targets = [i * 2 for i in data]

supervised_data = SupervisedData(data, targets)
(
    train_data,
    train_targets,
    test_data,
    test_targets,
) = supervised_data.get_train_and_test_data(batch_size=64, validation_ratio=0.2)

z1 = 0
a1 = 0
z2 = 0
a2 = 0


def forward(x):
    z1 = x @ w_input_hidden
    z1 += b_hidden
    
    a1 = sigmoid(z1)
    
    z2 = a1 @ w_hidden_output
    z2 += b_output
    
    a2 = sigmoid(z2)
    return a2

def backward(x, y):
    
