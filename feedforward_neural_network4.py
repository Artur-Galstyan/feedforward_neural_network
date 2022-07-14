import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from joblib import Memory

memory = Memory("./mnist")

np.random.seed(42)


def split_into_batches(x, batch_size):
    n_batches = len(x) / batch_size
    x = np.array_split(x, n_batches)
    return np.array(x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))


batch_size = 1

n_input = 784
n_hidden_1 = 64
n_output = 10

w_input_hidden = np.random.randn(n_input, n_hidden_1) * 0.1
w_hidden_output = np.random.randn(n_hidden_1, n_output) * 0.1

b_hidden = np.random.randn(n_hidden_1) * 0.1
b_output = np.random.randn(n_output) * 0.1

print("Downloading MNIST...")
fetch_openml_cached = memory.cache(fetch_openml)
mnist = fetch_openml_cached("mnist_784")

print("Download complete.")

X_train, X_test, y_train, y_test = train_test_split(
    mnist.data, mnist.target, test_size=0.33, random_state=42
)

minmax_scaler = MinMaxScaler()
ohe = OneHotEncoder()

X_train = split_into_batches(minmax_scaler.fit_transform(np.array(X_train)), batch_size)
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

# plt.imshow(X_train[0][0].reshape(28, 28))
# plt.title(str(np.argmax(y_train[0][0])))
# plt.show()

z1 = 0
a1 = 0
z2 = 0
a2 = 0


def forward(x):
    global z1, a1, z2, a2
    z1 = x @ w_input_hidden
    z1 += b_hidden

    a1 = sigmoid(z1)

    z2 = a1 @ w_hidden_output
    z2 += b_output

    a2 = sigmoid(z2)
    return a2


def backward(x, y):
    global w_hidden_output, b_output, w_input_hidden, b_hidden
    delta_2 = np.multiply(2 * (y - a2), 1)
    delta_w_hidden_output = a1.T @ delta_2
    delta_b_output = delta_2

    delta_1 = np.multiply(delta_2 @ w_hidden_output.T, sigmoid_deriv(z1))
    delta_w_input_hidden = x.T @ delta_1
    delta_b_hidden = delta_1

    return delta_w_hidden_output, delta_b_output, delta_w_input_hidden, delta_b_hidden


def update(
    learning_rate,
    delta_w_hidden_output,
    delta_b_output,
    delta_w_input_hidden,
    delta_b_hidden,
):
    global w_hidden_output, b_output, w_input_hidden, b_hidden
    w_hidden_output -= learning_rate * delta_w_hidden_output
    b_output -= learning_rate * delta_b_output.mean()

    w_input_hidden -= learning_rate * delta_w_input_hidden
    b_hidden -= learning_rate * delta_b_hidden.mean()


n_epochs = 50
learning_rate = 0.1
print("Starting training...")

for epoch in tqdm(range(n_epochs)):
    for x, y in zip(X_train, y_train):
        forward(x)
        (
            delta_w_hidden_output,
            delta_b_output,
            delta_w_input_hidden,
            delta_b_hidden,
        ) = backward(x, y)
        batch_size = len(x)
        update(
            learning_rate,
            delta_w_hidden_output / batch_size,
            delta_b_output / batch_size,
            delta_w_input_hidden / batch_size,
            delta_b_hidden / batch_size,
        )

correct = 0

for x, y in zip(X_test, y_test):
    prediction = forward(x)
    pred = np.argmax(prediction)

    actual = np.argmax(y)

    if pred == actual:
        correct += 1

print(f"Accuracy = {correct / len(y_test)}")
