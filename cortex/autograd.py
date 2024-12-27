from typing import TYPE_CHECKING, List, Optional, Set

import numpy as np

if TYPE_CHECKING:
    from .tensor import Tensor


class AutogradContext:
    def __init__(self, tensor: "Tensor"):
        self.tensor = tensor
        self.device_manager = tensor.device_manager

    def zero_grad(self):
        """Reset gradient to zeros"""
        self.tensor.gradient = self.device_manager.zeros(self.tensor.data.shape)

    def _reduce_gradient(self, gradient, from_shape, to_shape):
        """Reduce gradient to match target shape for broadcasting"""
        xp = self.device_manager._get_array_module()

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

    def build_graph(
        self,
        visited_tensors: Optional[Set["Tensor"]] = None,
        topo_order: Optional[List["Tensor"]] = None,
    ) -> List["Tensor"]:
        """Build computational graph in topological order"""
        if visited_tensors is None:
            visited_tensors = set()
        if topo_order is None:
            topo_order = []

        visited_tensors.add(self.tensor)
        for parent in self.tensor.parents:
            if parent not in visited_tensors:
                parent.autograd.build_graph(visited_tensors, topo_order)

        topo_order.append(self.tensor)
        return topo_order

    def backward(self, gradient=None):
        """Execute backward pass to compute gradients

        Args:
            gradient: Optional gradient to start backpropagation with.
                    Useful for broadcasting and non-scalar outputs.
        """
        topo_order = self.build_graph()
        xp = self.device_manager._get_array_module()

        if gradient is None:
            if self.tensor.data.shape == (1,) or self.tensor.data.shape == ():
                if xp.array_equal(
                    self.tensor.gradient,
                    self.device_manager.zeros(self.tensor.data.shape),
                ):
                    self.tensor.gradient = self.device_manager.ones(
                        self.tensor.data.shape
                    )
        else:
            self.tensor.gradient = self.device_manager.init_data(
                gradient.data, self.tensor.dtype
            )

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

            elif node.operation == "*":
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

            elif node.operation == "-":
                node.parents[0].gradient += self._reduce_gradient(
                    node.gradient, node.shape, node.parents[0].shape
                )
                node.parents[1].gradient += self._reduce_gradient(
                    -node.gradient, node.shape, node.parents[1].shape
                )

            elif node.operation == "/":
                node.parents[0].gradient += self._reduce_gradient(
                    node.gradient / node.parents[1].data,
                    node.shape,
                    node.parents[0].shape,
                )
                node.parents[1].gradient += self._reduce_gradient(
                    -node.gradient * node.parents[0].data / (node.parents[1].data ** 2),
                    node.shape,
                    node.parents[1].shape,
                )

            elif node.operation == "**":
                eps = 1e-7
                node.parents[0].gradient += self._reduce_gradient(
                    node.gradient
                    * node.parents[1].data
                    * (node.parents[0].data + eps) ** (node.parents[1].data - 1),
                    node.shape,
                    node.parents[0].shape,
                )
                node.parents[1].gradient += self._reduce_gradient(
                    node.gradient * node.data * xp.log(node.parents[0].data + eps),
                    node.shape,
                    node.parents[1].shape,
                )

            elif node.operation == "relu":
                grad = node.gradient * (node.parents[0].data > 0)
                node.parents[0].gradient += self._reduce_gradient(
                    grad, node.shape, node.parents[0].shape
                )

            elif node.operation == "tanh":
                node.parents[0].gradient += self._reduce_gradient(
                    node.gradient * (1 - node.data**2),
                    node.shape,
                    node.parents[0].shape,
                )

            elif node.operation == "sum":
                node.parents[0].gradient += self._reduce_gradient(
                    node.gradient * self.device_manager.ones(node.parents[0].shape),
                    node.shape,
                    node.parents[0].shape,
                )

            elif node.operation == "mean":
                size = float(np.prod(node.parents[0].shape))
                node.parents[0].gradient += self._reduce_gradient(
                    node.gradient
                    * self.device_manager.ones(node.parents[0].shape)
                    / size,
                    node.shape,
                    node.parents[0].shape,
                )

            elif node.operation == "reshape":
                node.parents[0].gradient += self._reduce_gradient(
                    xp.reshape(node.gradient, node.parents[0].shape),
                    node.shape,
                    node.parents[0].shape,
                )

            elif isinstance(node.operation, tuple) and node.operation[0] == "transpose":
                _, axes = node.operation
                node.parents[0].gradient += self._reduce_gradient(
                    xp.transpose(node.gradient, axes), node.shape, node.parents[0].shape
                )
