from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from .tensor import Tensor


class TensorOperations:
    def __init__(self, tensor):
        self.tensor = tensor
        self.device_manager = tensor.device_manager

    def _check_compatibility(self, other):
        """Check if other tensor is compatible for operations"""
        if not isinstance(other, type(self.tensor)):
            other = type(self.tensor)(
                other,
                device=self.tensor.device,
                dtype=self.tensor.dtype,
                requires_grad=False,  # Add default value
            )
        if self.tensor.device != other.device:
            raise ValueError(
                f"Cannot operate on different device types {self.tensor.device} and {other.device}"
            )
        if self.tensor.dtype != other.dtype:
            raise ValueError(
                f"Cannot operate on different dtypes {self.tensor.dtype} and {other.dtype}"
            )
        return other

    def _create_tensor(self, data, parents: list, operation: str) -> "Tensor":
        """Helper method to create a new tensor with same properties"""
        if len(parents) == 2:
            broadcast_shape = self.device_manager.broadcast_shape(
                parents[0].shape, parents[1].shape
            )
        else:
            broadcast_shape = data.shape

        return type(self.tensor)(
            data,
            device=self.tensor.device,
            dtype=self.tensor.dtype,
            gradient=self.device_manager.zeros(broadcast_shape),
            parents=parents,
            operation=operation,
        )

    # Arithmetic Operations
    def add(self, other):
        other = self._check_compatibility(other)
        return self._create_tensor(
            self.tensor.data + other.data, [self.tensor, other], "+"
        )

    def radd(self, other):
        return self.add(other)

    def multiply(self, other):
        other = self._check_compatibility(other)
        return self._create_tensor(
            self.tensor.data * other.data, [self.tensor, other], "*"
        )

    def rmultiply(self, other):
        return self.multiply(other)

    def subtract(self, other):
        other = self._check_compatibility(other)
        return self._create_tensor(
            self.tensor.data - other.data, [self.tensor, other], "-"
        )

    def rsubtract(self, other):
        other = self._check_compatibility(other)
        return self._create_tensor(
            other.data - self.tensor.data, [other, self.tensor], "-"
        )

    def divide(self, other):
        other = self._check_compatibility(other)
        if (other.data == 0).any():
            raise ValueError("Division by zero encountered")
        return self._create_tensor(
            self.tensor.data / other.data, [self.tensor, other], "/"
        )

    def rdivide(self, other):
        other = self._check_compatibility(other)
        if (self.tensor.data == 0).any():
            raise ValueError("Division by zero encountered")
        return self._create_tensor(
            other.data / self.tensor.data, [other, self.tensor], "/"
        )

    def power(self, other):
        other = self._check_compatibility(other)
        if (self.tensor.data <= 0).any():
            raise ValueError(
                "Cannot compute power of negative numbers - will cause issues in backward pass"
            )
        return self._create_tensor(
            self.tensor.data**other.data, [self.tensor, other], "**"
        )

    def rpower(self, other):
        other = self._check_compatibility(other)
        return self._create_tensor(
            other.data**self.tensor.data, [other, self.tensor], "**"
        )

    # Neural Network Operations
    def relu(self):
        xp = self.device_manager._get_array_module()
        output = xp.maximum(0, self.tensor.data)
        return self._create_tensor(output, [self.tensor], "relu")

    def tanh(self):
        xp = self.device_manager._get_array_module()
        return self._create_tensor(xp.tanh(self.tensor.data), [self.tensor], "tanh")

    # Tensor Operations
    def sum(self):
        xp = self.device_manager._get_array_module()
        return self._create_tensor(xp.sum(self.tensor.data), [self.tensor], "sum")

    def mean(self):
        xp = self.device_manager._get_array_module()
        return self._create_tensor(xp.mean(self.tensor.data), [self.tensor], "mean")

    def reshape(self, new_shape: Tuple[int, ...]):
        xp = self.device_manager._get_array_module()
        return self._create_tensor(
            xp.reshape(self.tensor.data, new_shape), [self.tensor], "reshape"
        )

    def transpose(self, *axes):
        xp = self.device_manager._get_array_module()
        axes = axes if axes else None
        return self._create_tensor(
            xp.transpose(self.tensor.data, axes), [self.tensor], ("transpose", axes)
        )
