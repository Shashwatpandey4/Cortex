import unittest

from cortex.tensor import Tensor


class TestTensor(unittest.TestCase):
    """
    Test suite for the Tensor class implementation.
    Verifies core functionality including arithmetic operations,
    gradient computation, and reduction operations.
    """

    def test_basic_operations(self):
        """
        Validates fundamental arithmetic operations on Tensor objects.
        Ensures proper implementation of addition, multiplication,
        subtraction, division, and power operations.
        """
        x = Tensor([2.0])
        y = Tensor([3.0])

        z = x + y
        self.assertEqual(z.data.tolist(), [5.0])

        z = x * y
        self.assertEqual(z.data.tolist(), [6.0])

        z = y - x
        self.assertEqual(z.data.tolist(), [1.0])

        z = y / x
        self.assertEqual(z.data.tolist(), [1.5])

        z = x ** Tensor([2.0])
        self.assertEqual(z.data.tolist(), [4.0])

    def test_more_complex_gradients(self):
        """
        Tests gradient computation for a complex composite function:
        f(w,x,y,z) = (w * x - y) / (z ** 2) + y * x

        This test validates the correct implementation of the chain rule
        and backpropagation through multiple arithmetic operations.

        The gradients are verified against manually calculated derivatives:
        ∂f/∂w = x/(z^2)
        ∂f/∂x = w/(z^2) + y
        ∂f/∂y = -1/(z^2) + x
        ∂f/∂z = -2(w*x - y)/(z^3)
        """
        w = Tensor([2.0])
        x = Tensor([3.0])
        y = Tensor([4.0])
        z = Tensor([2.0])

        a = w * x
        b = y
        c = a - b
        d = z**2
        e = c / d
        f = y * x
        out = e + f

        out.backward()

        self.assertAlmostEqual(w.gradient.item(), 3 / 4)
        self.assertAlmostEqual(x.gradient.item(), 4.5)
        self.assertAlmostEqual(y.gradient.item(), 2.75)
        self.assertAlmostEqual(z.gradient.item(), -0.5)

    def test_reduction_operations(self):
        """
        Validates reduction operations (sum and mean) and their gradients.

        Sum operation should propagate gradient of 1.0 to all input elements.
        Mean operation should propagate gradient of 1/n to all input elements,
        where n is the number of elements in the input tensor.
        """
        x = Tensor([1.0, 2.0, 3.0, 4.0])

        y = x.sum()
        self.assertEqual(y.data.item(), 10.0)
        y.backward()
        self.assertEqual(x.gradient.tolist(), [1.0, 1.0, 1.0, 1.0])

        x = Tensor([1.0, 2.0, 3.0, 4.0])
        y = x.mean()
        self.assertEqual(y.data.item(), 2.5)
        y.backward()
        self.assertEqual(x.gradient.tolist(), [0.25, 0.25, 0.25, 0.25])

    def test_error_handling(self):
        """
        Verifies proper error handling for invalid operations.
        Currently tests division by zero protection, but should be
        expanded to cover other edge cases and invalid operations
        as the Tensor implementation grows.
        """
        x = Tensor([1.0])
        y = Tensor([0.0])
        with self.assertRaises(ValueError):
            x / y


if __name__ == "__main__":
    unittest.main()
