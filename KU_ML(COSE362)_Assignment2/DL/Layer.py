import math
from abc import ABCMeta, abstractmethod

import numpy as np


# Base of Layer class.
# All layers have forward and backpropagation method, so it forces to implement these methods.
class LayerBase:
    def __init__(self):
        self.output = None  # for sigmoid, softmax, etc.
        self.input = None  # for backpropagation input

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.output = self.forward(x)
        self.input = x.copy()
        return self.output.copy()

    # method forward pass
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # method for backpropagation
    # It caculate new upstream for previous layer, with respect to upstream_grad and input x.
    @abstractmethod
    def backprop(self, x: np.ndarray, upstream_grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError


# Interface for layers that can update weights like LinearLayer.
# If class inherit this interface, it should implement update_weights method.
class AutoGradLayerInterface(metaclass=ABCMeta):
    @abstractmethod
    def update_weights(
        self, x: np.ndarray, upstream_grad: np.ndarray, lr: float
    ) -> np.ndarray:
        raise NotImplementedError


class LinearLayer(LayerBase, AutoGradLayerInterface):
    def __init__(self, input_size, output_size, std=1):
        self._input_size = input_size
        self._output_size = output_size
        super().__init__()

        # Xavier / He initialization
        self.W = np.random.normal(
            0, std / math.sqrt(input_size / 2), (output_size, input_size)
        )
        self.b = 0.01 + np.zeros((output_size, 1))

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Linear transformation
        score = self.W @ x.T + self.b
        return score.T

    def backprop(self, x: np.ndarray, upstream_grad: np.ndarray) -> np.ndarray:
        N = x.shape[0]  # batch_size (or number of data instances)
        """
        upstream_grad : gradient from previous layer
        We know that `grad_x = upstream_grad @ W`, but each data instance has different gradient and IID.
        So, we need to calculate (1,j) @ (j,k) matrix multiply for each data instance.
        - i : data instance index
        - j : output_size : for W
        - k : input_size : for x
        """
        grad_x = np.einsum("ij,ijk->ik", upstream_grad, np.array([self.W] * N))

        return grad_x

    # for update weights
    def update_weights(
        self, x: np.ndarray, upstream_grad: np.ndarray, lr: float
    ) -> np.ndarray:
        """
        Each data instance is IID, so cacaulate j @ k matrix mutiply for each data instance.
        - i : data instance index
        - j : output_size : for W, which makes linear transformmation (k -> j)
        - k : input_size : for x
        """
        grad_W = np.einsum("ij,ik->ijk", upstream_grad, x).mean(axis=0)

        """
        Likewise each data instance is IID.
        b(bias) is element-wise addition, so just sum all gradients and divide by batch_size.
        [:, None] is for shape (output_size, 1).
        """
        grad_b = upstream_grad.mean(axis=0)[:, None]

        self.W -= lr * grad_W  # update weights
        self.b -= lr * grad_b  # update bias


"""
- After this comments, we presupposes that each data instance is IID. 
- so we omits mention about matrix caculation based on IId in all comments.
- We also must consider upstream_grad as (batch_size, output_size) matrix, due to batch processing.
"""


class ReLU(LayerBase):
    def __init__(self):
        super().__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        # element-wise maximum between 0 and x
        return np.maximum(0, x)

    def backprop(self, x: np.ndarray, upstream_grad: np.ndarray) -> np.ndarray:
        """
        Element-wise gradient of ReLU function.
        So, upstream_grad dimension is same as x. Then, we can ultilize element-wise multiplication.
        """
        return upstream_grad * (x > 0)


class Sigmoid(LayerBase):
    def __init__(self):
        super().__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        # element-wise sigmoid function
        return 1 / (1 + np.exp(-x))

    def backprop(self, x: np.ndarray, upstream_grad: np.ndarray) -> np.ndarray:
        sigmoid = self.output
        # Likewise, we will use element-wise multiplication.
        return upstream_grad * sigmoid * (1 - sigmoid)


class Tanh(LayerBase):
    def __init__(self):
        super().__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        # element-wise sigmoid function
        return np.tanh(x)

    def backprop(self, x: np.ndarray, upstream_grad: np.ndarray) -> np.ndarray:
        tanh = self.output
        # Likewise, we will use element-wise multiplication.
        return upstream_grad * (1 - tanh**2)


class Softmax(LayerBase):
    def __init__(self):
        super().__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        For preventing overflow, we subtract max value from x.
        When dividing by sum of exps, we can get same result.
        """
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def backprop(self, x: np.ndarray, upstream_grad: np.ndarray) -> np.ndarray:
        softmax = self.output

        """
        - i : data instance index
        - j : output_size
        - k : output_size, same as j. Just for matching shape of Jacobian.

        - if j == k: grad_softmax[i,j,k] = softmax[i, j] * (1 - softmax[i, k])
            - softmax[i, j] * (1 - softmax[i, k]) = softmax[i, j] - softmax[i, j] * softmax[i, k]
            - we caculate case (j != k) and add softmax[i, j] from diagonal elements for fast compute.
        - if j != k: grad_softmax[i,j,k] = - softmax[i, j] * softmax[i, k]
        """
        grad_softmax = -np.einsum("ij,ik->ijk", softmax, softmax)
        diagnal_tensor = np.stack([np.diag(row) for row in softmax])
        grad_softmax += diagnal_tensor
        grad = np.einsum("ij,ijk->ik", upstream_grad, grad_softmax)

        return grad


class NNSequential(LayerBase):
    def __init__(self, *args):
        super().__init__()
        self.layers: list[LayerBase] = []
        for layer in args:
            if isinstance(layer, LayerBase):
                self.layers.append(layer)
            else:
                raise ValueError("All layers must be instance of LayerBase")

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer(x)
        return x

    def backprop(self, x: np.ndarray, upstream_grad: np.ndarray) -> np.ndarray:
        lr = float(x)
        reverse_layer = reversed(self.layers)
        for layer in reverse_layer:
            x = layer.input
            if isinstance(layer, AutoGradLayerInterface):
                layer.update_weights(x, upstream_grad, lr)
            upstream_grad = layer.backprop(x, upstream_grad)


if __name__ == "__main__":
    lr = 0.1
    model = NNSequential(LinearLayer(2, 3), ReLU(), LinearLayer(3, 4), Softmax())
    output = model.forward(np.array([[1, 2], [3, 4]]))
    # for softmax, sum of output is 1.
    print(output.sum(axis=1))
    print(output)
