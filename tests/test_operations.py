import unittest

import numpy as np

from cortex import Tensor


class TestTensorOperations(unittest.TestCase):
    def setUp(self):
        self.t1 = Tensor([1, 2, 3])
        self.t2 = Tensor([4, 5, 6])
        self.scalar = Tensor(2)

    def test_arithmetic_operations(self):
        """Test basic arithmetic operations"""
        # Addition
        result = self.t1 + self.t2
        np.testing.assert_array_equal(result.data, np.array([5, 7, 9]))

        # Multiplication
        result = self.t1 * self.t2
        np.testing.assert_array_equal(result.data, np.array([4, 10, 18]))

        # Subtraction
        result = self.t2 - self.t1
        np.testing.assert_array_equal(result.data, np.array([3, 3, 3]))

        # Division
        result = self.t2 / self.t1
        np.testing.assert_array_almost_equal(result.data, np.array([4, 2.5, 2]))

    def test_broadcasting(self):
        """Test broadcasting operations"""
        # Scalar-tensor operations
        result = self.t1 + self.scalar
        np.testing.assert_array_equal(result.data, np.array([3, 4, 5]))

        # Broadcasting with different shapes
        t3 = Tensor([[1, 2], [3, 4]])
        t4 = Tensor([1, 2])
        result = t3 + t4
        np.testing.assert_array_equal(result.data, np.array([[2, 4], [4, 6]]))

    def test_neural_ops(self):
        """Test neural network operations"""
        # ReLU
        t = Tensor([-1, 0, 1])
        result = t.relu()
        np.testing.assert_array_equal(result.data, np.array([0, 0, 1]))

        # Tanh
        result = t.tanh()
        np.testing.assert_array_almost_equal(
            result.data, np.array([-0.7615942, 0, 0.7615942]), decimal=6
        )
