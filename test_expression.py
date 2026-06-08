from engine import Value
import random
a = Value(1.0)
b = Value(2.0)
c = a / b
d = a ** b
e = c + d
f = e.tanh()
g = f.exp()
output = g
output.backward()
variable = [a,b,c,d,e,f,g,output]
for var in variable:
    print(var.grad)