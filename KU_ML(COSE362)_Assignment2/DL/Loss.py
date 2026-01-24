from abc import abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from DataLoader import BatchDataLoader, Dataset
from Layer import LinearLayer, NNSequential, ReLU, Softmax


class LossLayerBase:
    def __init__(self):
        self.output = None
        self.input = None

    def __call__(self, *args):
        self.output = self.forward(*args)
        self.input = args
        return self.output.copy()

    @abstractmethod
    def forward(self, x: np.ndarray, y: np.ndarray):
        raise NotImplementedError

    @abstractmethod
    def backprop(self, x: np.ndarray, y: np.ndarray):
        raise NotImplementedError


class CrossEntropyLoss(LossLayerBase):
    def __init__(self, epsilon=1e-12):
        super().__init__()
        self.epsilon = epsilon

    # y is assuemd to be one-hot encoded
    def forward(self, x: np.ndarray, y: np.ndarray):
        # if y is column vector, squeeze it.
        if y.ndim == 2:
            y = y.squeeze()

        # clip x to avoid log(0) due to numerical instability
        x = np.clip(x, self.epsilon, 1.0 - self.epsilon)
        return -np.mean(y * np.log(x))

    # y is assuemd to be one-hot encoded
    def backprop(self, x: np.ndarray, y: np.ndarray):
        if y.ndim == 2:
            y = y.squeeze()
        # clip x to avoid log(0) due to numerical instability
        x = np.clip(x, self.epsilon, 1.0 - self.epsilon)
        grad = -y / x
        # for batch, divide by batch size
        return grad / y.shape[0]


# for test
if __name__ == "__main__":
    data = np.random.randn(100, 16)
    labels = np.zeros((100, 2))
    labels[np.arange(100), np.random.randint(0, 2, 100)] = 1
    dataset = Dataset(data, labels)
    dataloader = BatchDataLoader(dataset, batch_size=10)
    lr = 0.1
    criterion = CrossEntropyLoss()
    model = NNSequential(LinearLayer(16, 8), ReLU(), LinearLayer(8, 2), Softmax())

    losses = []
    epochs = 10000
    for i in range(0, epochs + 1):
        for x, y in dataloader:
            pred = model(x)
            losses.append(criterion(pred, y))
            model.backprop(lr, criterion.backprop(pred, y))
        if i % 1000 == 0:
            print(f"{i}-step Loss: {losses[-1] * 100:.4f}%")

    plt.plot(losses)
    plt.show()
