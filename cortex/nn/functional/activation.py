import math
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ...tensor import Tensor


class ActivationFunctions:
    """
    Class containing neural network activation functions.
    Each method takes a tensor and returns a new tensor with the activation applied.
    """

    def __init__(self, tensor: "Tensor"):
        self.tensor = tensor
        self.device_manager = tensor.device_manager

    def _create_tensor(self, data, parents: list, operation: str) -> "Tensor":
        """Helper method to create a new tensor with same properties"""
        return type(self.tensor)(
            data,
            device=self.tensor.device,
            dtype=self.tensor.dtype,
            gradient=self.device_manager.zeros(data.shape),
            parents=parents,
            operation=operation,
        )

    def relu(self) -> "Tensor":
        """
        Applies the Rectified Linear Unit (ReLU) function element-wise.

        ReLU(x) = max(0, x)

        Returns:
            Tensor: A new tensor with the ReLU activation applied
        """
        xp = self.device_manager._get_array_module()
        output = xp.maximum(0, self.tensor.data)
        return self._create_tensor(output, [self.tensor], "relu")

    def leaky_relu(self, negative_slope: float = 0.01) -> "Tensor":
        """
        Applies the Leaky ReLU function element-wise.

        LeakyReLU(x) = max(0, x) + negative_slope * min(0, x)

        Args:
            negative_slope: Controls the angle of the negative slope. Default: 0.01

        Returns:
            Tensor: A new tensor with the Leaky ReLU activation applied
        """
        xp = self.device_manager._get_array_module()
        output = xp.where(
            self.tensor.data > 0, self.tensor.data, self.tensor.data * negative_slope
        )
        return self._create_tensor(
            output, [self.tensor], f"leaky_relu_{negative_slope}"
        )

    def gelu(self) -> "Tensor":
        """
        Applies the Gaussian Error Linear Unit (GELU) function element-wise.

        GELU(x) = x * Φ(x), where Φ(x) is the standard normal CDF
        This implementation uses the approximation:
        GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))

        Returns:
            Tensor: A new tensor with the GELU activation applied
        """
        xp = self.device_manager._get_array_module()
        x = self.tensor.data
        cdf = 0.5 * (
            1.0 + xp.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * xp.power(x, 3)))
        )
        output = x * cdf
        return self._create_tensor(output, [self.tensor], "gelu")

    def tanh(self) -> "Tensor":
        """
        Applies the hyperbolic tangent function element-wise.

        tanh(x) = (e^x - e^-x) / (e^x + e^-x)

        Returns:
            Tensor: A new tensor with the tanh activation applied
        """
        xp = self.device_manager._get_array_module()
        return self._create_tensor(xp.tanh(self.tensor.data), [self.tensor], "tanh")

    def sigmoid(self) -> "Tensor":
        """
        Applies the sigmoid function element-wise.

        sigmoid(x) = 1 / (1 + e^(-x))

        Returns:
            Tensor: A new tensor with the sigmoid activation applied
        """
        xp = self.device_manager._get_array_module()
        output = 1 / (1 + xp.exp(-self.tensor.data))
        return self._create_tensor(output, [self.tensor], "sigmoid")

    def softmax(self, dim: Optional[int] = None) -> "Tensor":
        """
        Applies the softmax function along a dimension.

        softmax(x_i) = exp(x_i) / sum(exp(x_j))

        Args:
            dim: Dimension along which to compute softmax. If None,
                 applies to flattened tensor.

        Returns:
            Tensor: A new tensor with the softmax activation applied
        """
        xp = self.device_manager._get_array_module()

        if dim is None:
            input_data = self.tensor.data.reshape(-1)
            dim = 0
        else:
            input_data = self.tensor.data

        max_vals = xp.max(input_data, axis=dim, keepdims=True)
        exp_x = xp.exp(input_data - max_vals)

        output = exp_x / xp.sum(exp_x, axis=dim, keepdims=True)

        if dim is None:
            output = output.reshape(self.tensor.shape)

        return self._create_tensor(output, [self.tensor], f"softmax_{dim}")

    def __call__(self, name: str, **kwargs) -> "Tensor":
        """
        Allows calling activation functions by name.

        Args:
            name: Name of the activation function
            **kwargs: Additional arguments for the activation function

        Returns:
            Tensor: Result of applying the specified activation function

        Raises:
            ValueError: If the activation function name is not recognized
        """
        activation_map = {
            "relu": self.relu,
            "leaky_relu": self.leaky_relu,
            "gelu": self.gelu,
            "tanh": self.tanh,
            "sigmoid": self.sigmoid,
            "softmax": self.softmax,
        }

        if name not in activation_map:
            raise ValueError(f"Unknown activation function: {name}")

        return activation_map[name](**kwargs)
