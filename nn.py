from engine import Value
import random
class Neuron():
    def __init__(self,nin,activation = True): ## nin : num of inputs
        self.w = [Value(random.uniform(-10,10)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))
        self.activation = activation
    def __call__(self,x): ## output = w * x + b
        ## shape (x : [1,len] , w : [1,len])
        out = sum(xi * wi for xi,wi in zip(x,self.w)) + self.b
        out = out.tanh() if self.activation else out
        return out
    def parameter(self):
        return self.w + [self.b]
class Layer():
    def __init__(self,nin,nout,activation = True): ## nin : 这一层的input个数,nout这一层的output的个数
        self.neurons = [Neuron(nin,activation) for _ in range(nout)]
    def __call__(self,x):
        out = [n(x) for n in self.neurons] ## 一层有几个node ?  --> 由nout决定,上一层几个node? --> 由nin决定!
        return out 
    def parameter(self):
        para = [p for n in self.neurons for p in n.parameter()]
        return para
class MLP():
    def __init__(self,nin,nout): ## nin是最初始的input的shape ,nout也就是后续的layer分别有几个neurons
        sz = [nin] + nout
        self.layer = [Layer(sz[i],sz[i + 1]) for i in range(len(nout) - 1)]
        self.layer.append(Layer(sz[-2],sz[-1],False))
    def __call__(self,x):
        for l in self.layer:
            x = l(x)
        return x
    def parameter(self):
        para = [p for n in self.layer for p in n.parameter()]
        return para