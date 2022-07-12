import numpy as np
from typing import Iterable


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


class SupervisedData:
    def __init__(self, data: np.iterable, targets: np.iterable) -> None:
        assert len(data) == len(
            targets
        ), f"Training data and targets must have the same count. {len(data)} != {len(targets)}"

        self.data = np.array(data)
        self.targets = np.array(targets)
        self.n_samples = len(data)

    def __len__(self):
        return self.n_samples

    def get_train_and_test_data(self, batch_size, validation_ratio):
        indices = list(range(len(self)))
        split = int(np.floor(validation_ratio * len(self)))

        train_indices, val_indices = indices[split:], indices[:split]
        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)
        train_indices = np.array(train_indices)
        val_indices = np.array(val_indices)
        train_data = split_into_batches(self.data[train_indices], batch_size)
        train_targets = split_into_batches(self.targets[train_indices], batch_size)
        test_data = split_into_batches(self.data[val_indices], batch_size)
        test_targets = split_into_batches(self.targets[val_indices], batch_size)

        return train_data, train_targets, test_data, test_targets


def split_into_batches(data, batch_size):
    temp = []
    batch_size = batch_size

    remainder = len(data) % batch_size

    i = 0
    while i < len(data):
        temp.append(data[i : i + batch_size])
        if i + batch_size <= len(data):
            i += batch_size
        else:
            i += remainder
    return temp


if __name__ == "__main__":
    batch_size = 3

    x = [np.random.randint(-5, 5) for i in range(10)]
    y = [i * 2 for i in x]

    x_train, x_target, y_test, y_target = SupervisedData(x, y).get_train_and_test_data(
        batch_size, 0.2
    )

    print(x_train, x_target)
    # print(y_test, y_target)
