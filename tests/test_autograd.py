import unittest

import numpy as np

from cortex import Tensor


class TestAutograd(unittest.TestCase):
    def setUp(self):
        self.x = Tensor([2.0], requires_grad=True)
        self.y = Tensor([3.0], requires_grad=True)

    def test_simple_backward(self):
        """Test simple backward pass"""
        z = self.x * self.y  # z = 6
        z.backward()

        self.assertEqual(self.x.gradient.item(), 3.0)  # dz/dx = y = 3
        self.assertEqual(self.y.gradient.item(), 2.0)  # dz/dy = x = 2

    def test_complex_backward(self):
        """Test more complex backward pass"""
        w = self.x * self.y  # w = 6
        z = w * self.x  # z = 12
        z.backward()

        # dz/dx = d(x^2*y)/dx = 2xy = 12
        self.assertEqual(self.x.gradient.item(), 12.0)
        # dz/dy = d(x^2*y)/dy = x^2 = 4
        self.assertEqual(self.y.gradient.item(), 4.0)

    def test_broadcast_backward(self):
        """Test backprop with broadcasting"""
        x = Tensor([[1, 2, 3]], requires_grad=True)
        y = Tensor([[1], [2]], requires_grad=True)
        z = x + y  # Broadcasting occurs
        z.backward(Tensor([[1, 1, 1], [1, 1, 1]]))

        np.testing.assert_array_equal(x.gradient.data, np.array([[2, 2, 2]]))
        np.testing.assert_array_equal(y.gradient.data, np.array([[3], [3]]))
