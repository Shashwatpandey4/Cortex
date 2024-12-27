import numpy as np

from .autograd import AutogradContext
from .device import DeviceManager
from .operations import TensorOperations


class Tensor:
    def __init__(
        self,
        data,
        device="cpu",
        dtype=np.float32,
        gradient=None,
        parents=None,
        operation="",
        requires_grad=False,
    ):
        self.device = device
        self.device_manager = DeviceManager(device)
        self.dtype = dtype
        self.data = self.device_manager.init_data(data, dtype)
        self.shape = self.data.shape
        self.gradient = (
            gradient if gradient is not None else self.device_manager.zeros(self.shape)
        )
        self.parents = parents if parents is not None else []
        self.operation = operation
        self.requires_grad = requires_grad

        self.ops = TensorOperations(self)
        self.autograd = AutogradContext(self)

    def __repr__(self):
        return f"Tensor(\n{self.data})"

    # Arithmetic operations
    def __add__(self, other):
        return self.ops.add(other)

    def __radd__(self, other):
        return self.ops.radd(other)

    def __mul__(self, other):
        return self.ops.multiply(other)

    def __rmul__(self, other):
        return self.ops.rmultiply(other)

    def __sub__(self, other):
        return self.ops.subtract(other)

    def __rsub__(self, other):
        return self.ops.rsubtract(other)

    def __truediv__(self, other):
        return self.ops.divide(other)

    def __rtruediv__(self, other):
        return self.ops.rdivide(other)

    def __pow__(self, other):
        return self.ops.power(other)

    def __rpow__(self, other):
        return self.ops.rpower(other)

    # Neural network operations
    def relu(self):
        return self.ops.relu()

    def tanh(self):
        return self.ops.tanh()

    # Tensor operations
    def sum(self):
        return self.ops.sum()

    def mean(self):
        return self.ops.mean()

    def reshape(self, new_shape):
        return self.ops.reshape(new_shape)

    def transpose(self, *axes):
        return self.ops.transpose(*axes)

    # Gradient operations
    def zero_grad(self):
        self.autograd.zero_grad()

    def backward(self, gradient=None):
        return self.autograd.backward(gradient)

    # Device management
    def to(self, device):
        return self.device_manager.transfer_to(device, self)
