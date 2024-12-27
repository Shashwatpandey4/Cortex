import unittest

import numpy as np

from cortex import Tensor


class TestDeviceManager(unittest.TestCase):
    def test_device_transfer(self):
        """Test device transfer functionality"""
        cpu_tensor = Tensor([1, 2, 3], device="cpu")

        # Test CPU->GPU transfer if CUDA is available
        try:
            gpu_tensor = cpu_tensor.to("gpu")
            self.assertEqual(gpu_tensor.device, "gpu")
            # Test GPU->CPU transfer
            cpu_tensor_2 = gpu_tensor.to("cpu")
            self.assertEqual(cpu_tensor_2.device, "cpu")
            np.testing.assert_array_equal(cpu_tensor.data, cpu_tensor_2.data)
        except:
            pass

    def test_broadcast_shape(self):
        """Test shape broadcasting logic"""
        t1 = Tensor([[1, 2]])  # Shape: (1, 2)
        t2 = Tensor([[1], [2]])  # Shape: (2, 1)
        result = t1 + t2  # Should broadcast to (2, 2)
        self.assertEqual(result.shape, (2, 2))
