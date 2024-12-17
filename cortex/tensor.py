import cupy as cp
import numpy as np


class Tensor:
    def __init__(
        self,
        data,
        device="cpu",
        dtype=np.float32,
        gradient=None,
        parents=None,
        operation="",
    ):
        self.device = device
        self.dtype = dtype
        self.data = self._init_data(data)
        self.shape = self.data.shape
        zeros = self._zeros(self.data.shape)
        self.gradient = gradient if gradient is not None else self._init_data(zeros)
        self.parents = parents if parents is not None else []
        self.operation = operation

    def _get_array_module(self):
        return cp if self.device == "gpu" else np

    def _log(self, x):
        xp = self._get_array_module()
        if (x <= 0).any():
            raise ValueError("Log of negative number encountered")
        return xp.log(x)

    def _power(self, x, power):
        xp = self._get_array_module()
        return xp.power(x, power)

    def _zeros(self, shape):
        xp = self._get_array_module()
        return xp.zeros(shape)

    def _ones(self, shape):
        xp = self._get_array_module()
        return xp.ones(shape)

    def _init_data(self, data, dtype=None):
        xp = self._get_array_module()
        if isinstance(data, (list, tuple)):
            data = xp.array(data, dtype=self.dtype)
        elif isinstance(data, (int, float, np.number)):
            data = xp.array([data], dtype=self.dtype)
        elif not isinstance(data, (xp.ndarray)):
            raise TypeError(f"Unsupported data type: {type(data)}")
        return data

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
        broadcast_shape = self._broadcast_shape(self.shape, other.shape)
        return Tensor(
            self.data + other.data,
            device=self.device,
            dtype=self.dtype,
            gradient=self._zeros(broadcast_shape),
            parents=[self, other],
            operation="+",
        )

    def __radd__(self, other):
        other = self._check_compatibility(other)
        broadcast_shape = self._broadcast_shape(self.shape, other.shape)
        return Tensor(
            self.data + other.data,
            device=self.device,
            dtype=self.dtype,
            gradient=self._zeros(broadcast_shape),
            parents=[self, other],
            operation="+",
        )

    def __mul__(self, other):
        other = self._check_compatibility(other)
        broadcast_shape = self._broadcast_shape(self.shape, other.shape)
        return Tensor(
            self.data * other.data,
            device=self.device,
            dtype=self.dtype,
            gradient=self._zeros(broadcast_shape),
            parents=[self, other],
            operation="*",
        )

    def __rmul__(self, other):
        other = self._check_compatibility(other)
        broadcast_shape = self._broadcast_shape(self.shape, other.shape)
        return Tensor(
            self.data * other.data,
            device=self.device,
            dtype=self.dtype,
            gradient=self._zeros(broadcast_shape),
            parents=[self, other],
            operation="*",
        )

    def __sub__(self, other):
        other = self._check_compatibility(other)
        broadcast_shape = self._broadcast_shape(self.shape, other.shape)
        return Tensor(
            self.data - other.data,
            device=self.device,
            dtype=self.dtype,
            gradient=self._zeros(broadcast_shape),
            parents=[self, other],
            operation="-",
        )

    def __rsub__(self, other):
        other = self._check_compatibility(other)
        broadcast_shape = self._broadcast_shape(self.shape, other.shape)
        return Tensor(
            other.data - self.data,
            device=self.device,
            dtype=self.dtype,
            gradient=self._zeros(broadcast_shape),
            parents=[self, other],
            operation="-",
        )

    def __truediv__(self, other):
        other = self._check_compatibility(other)
        if (other.data == 0).any():
            raise ValueError("Division by zero encountered")
        broadcast_shape = self._broadcast_shape(self.shape, other.shape)
        return Tensor(
            self.data / other.data,
            device=self.device,
            dtype=self.dtype,
            gradient=self._zeros(broadcast_shape),
            parents=[self, other],
            operation="/",
        )

    def __rtruediv__(self, other):
        other = self._check_compatibility(other)
        broadcast_shape = self._broadcast_shape(self.shape, other.shape)
        return Tensor(
            other.data / self.data,
            device=self.device,
            dtype=self.dtype,
            gradient=self._zeros(broadcast_shape),
            parents=[self, other],
            operation="/",
        )

    def __pow__(self, other):
        other = self._check_compatibility(other)
        if (self.data <= 0).any():
            raise ValueError(
                "Cannot compute power of negative numbers - will cause issues in backward pass"
            )
        broadcast_shape = self._broadcast_shape(self.shape, other.shape)
        return Tensor(
            self.data**other.data,
            device=self.device,
            dtype=self.dtype,
            gradient=self._zeros(broadcast_shape),
            parents=[self, other],
            operation="**",
        )

    def __rpow__(self, other):
        other = self._check_compatibility(other)
        broadcast_shape = self._broadcast_shape(self.shape, other.shape)
        return Tensor(
            other.data**self.data,
            device=self.device,
            dtype=self.dtype,
            gradient=self._zeros(broadcast_shape),
            parents=[other, self],
            operation="**",
        )

    def sum(self):
        xp = self._get_array_module()
        return Tensor(
            xp.sum(self.data),
            device=self.device,
            dtype=self.dtype,
            parents=[self],
            operation="sum",
        )

    def mean(self):
        xp = self._get_array_module()
        return Tensor(
            xp.mean(self.data),
            device=self.device,
            dtype=self.dtype,
            parents=[self],
            operation="mean",
        )

    def relu(self):
        xp = self._get_array_module()
        output = xp.maximum(0, self.data)

        return Tensor(
            output,
            device=self.device,
            dtype=self.dtype,
            parents=[self],
            operation="relu",
            gradient=self._zeros(output.shape),
        )

    def tanh(self):
        xp = self._get_array_module()
        return Tensor(
            xp.tanh(self.data),
            device=self.device,
            dtype=self.dtype,
            parents=[self],
            operation="tanh",
        )

    def zero_grad(self):
        zeros = self._zeros(self.data.shape)
        self.gradient = self._init_data(zeros)

    def _broadcast_shape(self, shape1, shape2):
        xp = self._get_array_module()
        s1 = xp.array(shape1)
        s2 = xp.array(shape2)

        if len(s1) < len(s2):
            s1 = xp.pad(s1, (len(s2) - len(s1), 0), constant_values=1)
        elif len(s2) < len(s1):
            s2 = xp.pad(s2, (len(s1) - len(s2), 0), constant_values=1)

        if not all((s1 == s2) | (s1 == 1) | (s2 == 1)):
            raise ValueError(f"Shapes {shape1} and {shape2} are not broadcastable")

        return tuple(xp.maximum(s1, s2))

    def reshape(self, new_shape):
        xp = self._get_array_module()
        return Tensor(
            xp.reshape(self.data, new_shape),
            device=self.device,
            dtype=self.dtype,
            parents=[self],
            operation="reshape",
        )

    def transpose(self, *axes):
        xp = self._get_array_module()
        axes = axes if axes else None
        return Tensor(
            xp.transpose(self.data, axes),
            device=self.device,
            dtype=self.dtype,
            parents=[self],
            operation=("transpose", axes),
        )

    def _reduce_gradient(self, gradient, from_shape, to_shape):
        xp = self._get_array_module()

        if not from_shape:
            from_shape = (1,)
        if not to_shape:
            to_shape = (1,)

        if len(from_shape) > len(to_shape):
            sum_axis = tuple(range(len(from_shape) - len(to_shape)))
            gradient = xp.sum(gradient, axis=sum_axis)

        elif len(from_shape) == len(to_shape):
            sum_axis = []
            for i, (f, t) in enumerate(zip(from_shape, to_shape)):
                if f != t and t == 1:
                    sum_axis.append(i)
            if sum_axis:
                gradient = xp.sum(gradient, axis=tuple(sum_axis), keepdims=True)

        return gradient

    def backward(self):
        topo_order = self.build_graph()

        xp = self._get_array_module()
        if self.data.shape == (1,) or self.data.shape == ():
            if xp.array_equal(self.gradient, self._zeros(self.data.shape)):
                self.gradient = self._ones(self.data.shape)

        for node in reversed(topo_order):
            assert (
                node.gradient.shape == node.data.shape
            ), f"Gradient shape {node.gradient.shape} doesn't match data shape {node.data.shape}"
            if node.operation == "+":
                node.parents[0].gradient += self._reduce_gradient(
                    node.gradient, node.shape, node.parents[0].shape
                )
                node.parents[1].gradient += self._reduce_gradient(
                    node.gradient, node.shape, node.parents[1].shape
                )

            if node.operation == "*":
                node.parents[0].gradient += self._reduce_gradient(
                    node.parents[1].data * node.gradient,
                    node.shape,
                    node.parents[0].shape,
                )
                node.parents[1].gradient += self._reduce_gradient(
                    node.parents[0].data * node.gradient,
                    node.shape,
                    node.parents[1].shape,
                )

            if node.operation == "-":
                node.parents[0].gradient += self._reduce_gradient(
                    node.gradient, node.shape, node.parents[0].shape
                )
                node.parents[1].gradient += self._reduce_gradient(
                    -node.gradient, node.shape, node.parents[1].shape
                )

            if node.operation == "/":
                node.parents[0].gradient += self._reduce_gradient(
                    (1 / node.parents[1].data) * node.gradient,
                    node.shape,
                    node.parents[0].shape,
                )
                node.parents[1].gradient += self._reduce_gradient(
                    (-node.parents[0].data / self._power(node.parents[1].data, 2))
                    * node.gradient,
                    node.shape,
                    node.parents[1].shape,
                )

            if node.operation == "**":
                eps = 1e-7
                node.parents[0].gradient += self._reduce_gradient(
                    node.parents[1].data
                    * (
                        self._power(
                            node.parents[0].data + eps, node.parents[1].data - 1
                        )
                    )
                    * node.gradient,
                    node.shape,
                    node.parents[0].shape,
                )
                node.parents[1].gradient += self._reduce_gradient(
                    (self._power(node.parents[0].data, node.parents[1].data))
                    * self._log(node.parents[0].data)
                    * node.gradient,
                    node.shape,
                    node.parents[1].shape,
                )
            if node.operation == "relu":
                xp = self._get_array_module()
                mask = (node.parents[0].data > 0).astype(self.dtype)
                grad = node.gradient * mask
                node.parents[0].gradient += self._reduce_gradient(
                    grad, node.shape, node.parents[0].shape
                )

            if node.operation == "tanh":
                node.parents[0].gradient += self._reduce_gradient(
                    (1 - self._power(node.data, 2)) * node.gradient,
                    node.shape,
                    node.parents[0].shape,
                )
            if node.operation == "sum":
                node.parents[0].gradient += self._reduce_gradient(
                    node.gradient * self._ones(node.parents[0].shape),
                    node.shape,
                    node.parents[0].shape,
                )

            if node.operation == "mean":
                size = float(np.prod(node.parents[0].shape))
                node.parents[0].gradient += self._reduce_gradient(
                    node.gradient * self._ones(node.parents[0].shape) / size,
                    node.shape,
                    node.parents[0].shape,
                )

            if node.operation == "reshape":
                node.parents[0].gradient += self._reduce_gradient(
                    xp.reshape(node.gradient, node.parents[0].shape),
                    node.shape,
                    node.parents[0].shape,
                )

            if isinstance(node.operation, tuple) and node.operation[0] == "transpose":
                _, axes = node.operation
                node.parents[0].gradient += self._reduce_gradient(
                    xp.transpose(node.gradient, axes), node.shape, node.parents[0].shape
                )

    def build_graph(self, visited_tensors=None, topo_order=None):
        if visited_tensors is None:
            visited_tensors = set()
        if topo_order is None:
            topo_order = []

        visited_tensors.add(self)
        for parent in self.parents:
            if parent not in visited_tensors:
                parent.build_graph(visited_tensors, topo_order)

        topo_order.append(self)

        return topo_order

    def to(self, device):
        if device == self.device:
            return self

        if device == "cpu":
            new_data = cp.asnumpy(self.data) if self.device == "gpu" else self.data
            new_gradient = (
                cp.asnumpy(self.gradient) if self.device == "gpu" else self.gradient
            )
        elif device == "gpu":
            new_data = cp.array(self.data) if self.device == "cpu" else self.data
            new_gradient = (
                cp.array(self.gradient) if self.device == "cpu" else self.gradient
            )
        else:
            raise ValueError(f"Unknown device: {device}")

        return Tensor(
            new_data,
            device=device,
            dtype=self.dtype,
            gradient=new_gradient,
            parents=self.parents,
            operation=self.operation,
        )
