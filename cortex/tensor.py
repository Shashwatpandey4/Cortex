import cupy as cp
import numpy as np


class Tensor:
    def __init__(
        self,
        data,
        device="cpu",
        dtype=np.float32,
        parents=None,
        operation="",
        ctx=None,
    ):
        self.data = self._init_data(
            data,
            dtype,
            device,
        )
        self.device = device
        self.dtype = dtype
        self.shape = self.data.shape
        zeros = np.zeros(self.data.shape)
        self.gradient = self._init_data(zeros, dtype, device)
        self.parents = parents if parents is not None else []
        self.operation = operation
        self.ctx = ctx

    def _init_data(self, data, dtype, device):
        if device == "cpu":
            return np.array(data, dtype=dtype)
        elif device == "gpu":
            return cp.array(data, dtype=dtype)
        else:
            raise ValueError(f"Unkown device : {device}")

    def _check_compatibility(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, device=self.device, dtype=self.dtype)
        if self.device != other.device:
            raise ValueError(
                f"Cannot operate on different device types {self.device} and {other.device}"
            )
        if self.dtype != other.dtype:
            raise ValueError(
                f"Cannot operate on different dtypes {self.dtype} and {other.dtype}"
            )
        return other

    def __repr__(self):
        return f"Tensor(\n{self.data})"

    def __add__(self, other):
        other = self._check_compatibility(other)
        return Tensor(
            self.data + other.data,
            device=self.device,
            dtype=self.dtype,
            gradient=np.zeros(self.data.shape),
            parents=[self, other],
            operation="+",
        )

    def __radd__(self, other):
        other = self._check_compatibility(other)
        return Tensor(
            self.data + other.data,
            device=self.device,
            dtype=self.dtype,
            gradient=np.zeros(self.data.shape),
            parents=[self, other],
            operation="+",
        )

    def __mul__(self, other):
        other = self._check_compatibility(other)
        return Tensor(
            self.data * other.data,
            device=self.device,
            dtype=self.dtype,
            gradient=np.zeros(self.data.shape),
            parents=[self, other],
            operation="*",
        )

    def __rmul__(self, other):
        other = self._check_compatibility(other)
        return Tensor(
            self.data * other.data,
            device=self.device,
            dtype=self.dtype,
            gradient=np.zeros(self.data.shape),
            parents=[self, other],
            operation="*",
        )

    def __sub__(self, other):
        other = self._check_compatibility(other)
        return Tensor(
            self.data - other.data,
            device=self.device,
            dtype=self.dtype,
            gradient=np.zeros(self.data.shape),
            parents=[self, other],
            operation="-",
        )

    def __rsub__(self, other):
        other = self._check_compatibility(other)
        return Tensor(
            other.data - self.data,
            device=self.device,
            dtype=self.dtype,
            gradient=np.zeros(self.data.shape),
            parents=[self, other],
            operation="-",
        )

    def __truediv__(self, other):
        other = self._check_compatibility(other)
        return Tensor(
            self.data / other.data,
            device=self.device,
            dtype=self.dtype,
            gradient=np.zeros(self.data.shape),
            parents=[self, other],
            operation="/",
        )

    def __rtruediv__(self, other):
        other = self._check_compatibility(other)
        return Tensor(
            other.data / self.data,
            device=self.device,
            dtype=self.dtype,
            gradient=np.zeros(self.data.shape),
            parents=[self, other],
            operation="/",
        )

    def __pow__(self, other):
        other = self._check_compatibility(other)
        return Tensor(
            self.data**other.data,
            device=self.device,
            dtype=self.dtype,
            gradient=np.zeros(self.data.shape),
            parents=[self, other],
            operation="**",
        )

    def __rpow__(self, other):
        other = self._check_compatibility(other)
        return Tensor(
            self.data**other.data,
            device=self.device,
            dtype=self.dtype,
            gradient=np.zeros(self.data.shape),
            parents=[self, other],
            operation="**",
        )

    def zero_grad(self):
        zeros = np.zeros(self.data.shape)
        self.gradient = self._init_data(zeros, self.dtype, self.device)

    def backward(self):
        if self.data.shape == (1,) or self.data.shape == ():
            if self.device == "cpu" and np.array_equal(
                self.gradient, np.zeros(self.data.shape)
            ):
                ones = np.ones(self.data.shape)
                self.gradient = self._init_data(ones, self.dtype, self.device)

            if self.device == "gpu" and cp.array_equal(
                self.gradient, cp.zeros(self.data.shape)
            ):
                ones = cp.ones(self.data.shape)
                self.gradient = self._init_data(ones, self.dtype, self.device)

        if self.operation == "+":
            self.parents[0].gradient += 1 * self.gradient
            self.parents[1].gradient += 1 * self.gradient
            for parent in self.parents:
                if parent.operation:
                    parent.backward()

        if self.operation == "*":
            self.parents[0].gradient += 1 * self.parents[1].data * self.gradient
            self.parents[1].gradient += 1 * self.parents[0].data * self.gradient
            for parent in self.parents:
                if parent.operation:
                    parent.backward()

        if self.operation == "-":
            self.parents[0].gradient += 1 * self.gradient
            self.parents[1].gradient += -1 * self.gradient
            for parent in self.parents:
                if parent.operation:
                    parent.backward()

        if self.operation == "/":
            self.parents[0].gradient += (1 / self.parents[1].data) * self.gradient
            self.parents[1].gradient += (
                (-self.parents[0].data) / (self.parents[1].data ** 2) * self.gradient
            )
            for parent in self.parents:
                if parent.operation:
                    parent.backward()

        if self.operation == "**":
            self.parents[0].gradient += (
                self.parents[1].data
                * (self.parents[0].data ** (self.parents[1].data - 1))
                * self.gradient
            )
            self.parents[1].gradient += (
                (self.parents[0].data ** self.parents[1].data)
                * np.log(self.parents[0].data)
                * self.gradient
            )
            for parent in self.parents:
                if parent.operation:
                    parent.backward()
