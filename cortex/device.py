from typing import TYPE_CHECKING, Any, Tuple, Union

import cupy as cp
import numpy as np

if TYPE_CHECKING:
    from .tensor import Tensor


class DeviceManager:
    def __init__(self, device: str):
        if device not in ["cpu", "gpu"]:
            raise ValueError(f"Unsupported device: {device}")
        self.device = device

    def _get_array_module(self):
        """Get the appropriate array module (numpy/cupy) based on device"""
        return cp if self.device == "gpu" else np

    def init_data(self, data: Any, dtype: np.dtype) -> Union[np.ndarray, cp.ndarray]:
        """Initialize data with appropriate type and device"""
        xp = self._get_array_module()

        if isinstance(data, (list, tuple)):
            data = xp.array(data, dtype=dtype)
        elif isinstance(data, (int, float, np.number)):
            data = xp.array([data], dtype=dtype)
        elif not isinstance(data, (np.ndarray, cp.ndarray)):
            raise TypeError(f"Unsupported data type: {type(data)}")

        return data

    def zeros(self, shape: Tuple[int, ...]) -> Union[np.ndarray, cp.ndarray]:
        """Create zero tensor with given shape"""
        xp = self._get_array_module()
        return xp.zeros(shape)

    def ones(self, shape: Tuple[int, ...]) -> Union[np.ndarray, cp.ndarray]:
        """Create ones tensor with given shape"""
        xp = self._get_array_module()
        return xp.ones(shape)

    def broadcast_shape(
        self, shape1: Tuple[int, ...], shape2: Tuple[int, ...]
    ) -> Tuple[int, ...]:
        """Compute broadcast shape for two tensor shapes"""
        xp = self._get_array_module()
        s1 = xp.array(shape1)
        s2 = xp.array(shape2)

        # Pad shapes with ones to match lengths
        if len(s1) < len(s2):
            s1 = xp.pad(s1, (len(s2) - len(s1), 0), constant_values=1)
        elif len(s2) < len(s1):
            s2 = xp.pad(s2, (len(s1) - len(s2), 0), constant_values=1)

        # Check if shapes are broadcastable
        if not all((s1 == s2) | (s1 == 1) | (s2 == 1)):
            raise ValueError(f"Shapes {shape1} and {shape2} are not broadcastable")

        return tuple(xp.maximum(s1, s2))

    def transfer_to(self, target_device: str, tensor: "Tensor") -> "Tensor":
        """Transfer tensor to specified device"""
        if target_device == tensor.device:
            return tensor

        if target_device == "cpu":
            new_data = (
                cp.asnumpy(tensor.data) if tensor.device == "gpu" else tensor.data
            )
            new_gradient = (
                cp.asnumpy(tensor.gradient)
                if tensor.device == "gpu"
                else tensor.gradient
            )
        elif target_device == "gpu":
            if not cp.cuda.is_available():
                raise RuntimeError("CUDA is not available for GPU operations")
            new_data = cp.array(tensor.data) if tensor.device == "cpu" else tensor.data
            new_gradient = (
                cp.array(tensor.gradient) if tensor.device == "cpu" else tensor.gradient
            )
        else:
            raise ValueError(f"Unknown device: {target_device}")

        return type(tensor)(
            new_data,
            device=target_device,
            dtype=tensor.dtype,
            gradient=new_gradient,
            parents=tensor.parents,
            operation=tensor.operation,
        )
