
from engine import Value
from nn import *
import random

def update():
    loss = Value(0.0)
    output = n(x)
    for i in range(target_len):
        loss += (output[i] - Value(target[i])) ** 2
    para = n.parameter()
    for p in para:
        p.grad = 0
    loss.backward()
    for p in para:
        p.data -= lr * p.grad
    print(output)
    print('----------------')
    print(loss)

input_len = 4
x = [Value(random.uniform(-10,10)) for _ in range(4)]
target = [1,100,-100,-100]
n = MLP(4,[16,16,4])
target_len = 4
lr = 0.01

epoch = 50
for i in range(epoch):
    update()
