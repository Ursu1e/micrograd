from engine import Value
a = Value(2.0)
b = Value(3.0)
c = a * b
c.backward()
print(c._prev)
print(a.grad,b.grad,c.grad)