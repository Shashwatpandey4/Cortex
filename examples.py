from cortex.tensor import Tensor

a = Tensor(2)
b = Tensor(3)

print(a)
print(b)

c = 2 * a + b

print(f"c : {c}")
c.backward()

print(a.gradient)
print(b.gradient)
