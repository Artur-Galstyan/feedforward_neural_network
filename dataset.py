import numpy as np
from typing import Iterable


class SupervisedData:
    def __init__(self, data: Iterable, targets: Iterable) -> None:
        assert len(data) == len(
            targets
        ), f"Training data and targets must have the same count. {len(data)} != {len(targets)}"

        self.data = data
        self.targets = targets
        self.n_samples = len(data)

    def __len__(self):
        return self.n_samples

    def get_train_and_test_samplers(self, batch_size, validation_ratio):
        indices = list(range(len(self)))
        split = int(np.floor(validation_ratio * len(self)))

        train_indices, val_indices = indices[split:], indices[:split]
        return 0


if __name__ == "__main__":
    x = [1, 2, 3, 4]
    y = [2, 4, 6, 8]

    data = SupervisedData(x, y)
    data.get_train_and_test_samplers(2, 0.5)
