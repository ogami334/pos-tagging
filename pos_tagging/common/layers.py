import numpy as np
from .utils import softmax, cross_entropy_error
# from utils import softmaxだと動かなかった(他のファイルから呼ぶときは相対importしないとダメってこと？)


# Sigmoid function
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx


# Rectified Linear Unit
class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


# layers which integrate Softmax func and Cross Entropy Loss
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx


# Fully-Connected Layer
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx
# backwardを呼ぶと、self.dWとself.dbに勾配が保存される


class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None
        self.dW = None

    def forward(self, idx):
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out

    def backward(self, dout):
        self.dW, = self.grads
        self.dW[...] = 0
        np.add.at(self.dW, self.idx, dout)
        return self.dW
# backwardを呼ぶと、dWに勾配が保存される。


class SumLine:
    def __init__(self):
        self.shape = None

    def forward(self, x):
        self.shape = x.shape[1], x.shape[0]
        out = np.sum(x, axis=1)
        # xの行を全て合計した行ベクトルを返す
        return out

    def backward(self, dout):
        A = np.ones(shape=(self.shape), dtype='int32')
        dh = np.dot(A, dout)
        return dh
