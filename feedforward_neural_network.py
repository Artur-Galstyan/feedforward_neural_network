import numpy as np
import torch_helpers.TorchDataset as td


n_samples = 200
n_epochs = 500


simple_x = [np.random.uniform(-2, 2) for _ in range(n_samples)]
simple_y = [i * 2 + np.random.uniform(-0.05, 0.05) for i in simple_x]


training_set, test_set = td.TorchDataset(
    simple_x, simple_y
).get_train_and_validation_data(batch_size=1, validation_ratio=0.2)


w1 = np.random.randn()
b1 = np.random.randn()

w2 = np.random.randn()
b2 = np.random.randn()

learning_rate = 0.01
for epoch in range(n_epochs):
    avg_cost = 0

    for x, y in training_set:
        x = x.item()
        y = y.item()

        # Forward

        a1 = x * w1 + b1
        a2 = a1 * w2 + b2

        avg_cost += ((a2 - y) ** 2) / n_samples

        dc_da2 = 2 * (a2 - y)

        dc_dw2 = dc_da2 * a1
        dc_db2 = dc_da2

        dc_dw1 = (dc_da2 * w2) * x
        dc_db1 = dc_da2 * w2

    w1 -= learning_rate * dc_dw1
    b1 -= learning_rate * dc_db1

    w2 -= learning_rate * dc_dw2
    b2 -= learning_rate * dc_db1

    print(f"Avg. Cost for Epoch {epoch} = {avg_cost}")


for x, y in test_set:
    x = x.item()
    y = y.item()

    a1 = x * w1 + b1
    a2 = a1 * w2 + b2

    print(f"x = {x}; Expected {y}; Got {a2}")
