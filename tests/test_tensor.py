import unittest

import numpy as np

from cortex import Tensor


class TestTensorBasics(unittest.TestCase):
    def setUp(self):
        self.data = [1, 2, 3, 4]
        self.tensor = Tensor(self.data)

    def test_initialization(self):
        """Test tensor initialization with different data types"""
        # List initialization
        t1 = Tensor([1, 2, 3])
        self.assertEqual(t1.shape, (3,))

        # Scalar initialization
        t2 = Tensor(5)
        self.assertEqual(t2.shape, (1,))

        # Numpy array initialization
        t3 = Tensor(np.array([1, 2, 3]))
        self.assertEqual(t3.shape, (3,))

    def test_device_placement(self):
        """Test tensor device placement"""
        cpu_tensor = Tensor([1, 2, 3], device="cpu")
        self.assertEqual(cpu_tensor.device, "cpu")

        # Only test GPU if CUDA is available
        try:
            gpu_tensor = Tensor([1, 2, 3], device="gpu")
            self.assertEqual(gpu_tensor.device, "gpu")
        except:
            pass

    def test_invalid_initialization(self):
        """Test invalid tensor initializations"""
        with self.assertRaises(TypeError):
            Tensor("invalid data")

        with self.assertRaises(ValueError):
            Tensor([1, 2, 3], device="invalid_device")
