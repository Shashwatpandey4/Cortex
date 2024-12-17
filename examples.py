from cortex.tensor import Tensor

w = Tensor([2.0], device="cpu")
x = Tensor([3.0], device="cpu")
y = Tensor([4.0], device="cpu")
z = Tensor([2.0], device="cpu")

# Build graph step by step
a = w * x  # a = 6
b = y  # b = 4
c = a - b  # c = 2
d = z**2  # d = 4
e = c / d  # e = 0.5
f = y * x  # f = 12
out = e + f  # out = 12.5

out.backward()

print("Results:")
print(f"w gradient: {w.gradient}")  # Should be 3/4
print(f"x gradient: {x.gradient}")  # Should be 2/4 + 4
print(f"y gradient: {y.gradient}")  # Should be -1/4 + 3
print(f"z gradient: {z.gradient}")  # Should be -2 * (w*x - y)/(z**3)
