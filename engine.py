import math
class Value:
    def __init__(self,data,_child = (),op = ''):
        self.data = data
        self._prev = set(_child)
        self.op = op
        self.grad = 0.0
        self._backward = lambda : None
    def __add__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data + other.data,(self,other),'+')
        def _backward(): 
            self.grad += out.grad * 1.0
            other.grad += out.grad * 1.0
        out._backward = _backward
        return out
    def __mul__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data * other.data,(self,other),'*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    def exp(self):
        out = Value(math.exp(self.data),(self,),'exp')
        def _backward():
            self.grad += out.grad * math.exp(self.data)
        out._backward = _backward
        return out
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for child in reversed(topo):
            child._backward()
    def __pow__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        p = other.data
        out = Value(math.pow(self.data,p),(self,other),'**')
        def _backward():
            self.grad += p * math.pow(self.data,p - 1) * out.grad
        out._backward = _backward
        return out
    def tanh(self):
        data = self.data
        e = math.exp(2 * data)
        out = Value((e - 1) / (e + 1),(self,),'tanh')
        def _backward():
            self.grad += (1 - out.data ** 2) * out.grad
        out._backward = _backward
        return out
    def __truediv__(self,other):
        return self * (other ** -1)
    def __radd__(self,other):
        return self + other
    def __rmul__(self,other):
        return self * other
    def __sub__(self,other):
        return self + (-other)
    def __rsub__(self,other):
        return (-self) + other
    def __neg__(self):
        return self * Value(-1)
    def __repr__(self):
        return f'Value : {self.data} , grad : {self.grad}'
    