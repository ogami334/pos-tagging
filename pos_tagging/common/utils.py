import numpy as np
def softmax(a):
    c = np.max(a, axis=1, keepdims=True)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a, axis=1, keepdims=True)
    y = exp_a / sum_exp_a
    return y

# def cross_entropy_error(y, t):
#     if y.ndim == 1:
#         t = t.reshape(1, t.size)
#         y = y.reshape(1, y.size)

#     batch_size = y.shape[0]
#     return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size


# use case
# x = np.array([
#     [1, 2, 3],
#     [4, 5, 6]])
# mat_convert(x, 4)
# result:
# [[[1 2 3]
#   [1 2 3]
#   [1 2 3]
#   [1 2 3]]

#  [[4 5 6]
#   [4 5 6]
#   [4 5 6]
#   [4 5 6]]]
def mat_convert(x, s):
    hoge = []
    y = np.empty(shape=(s, 3), dtype="int")
    for i in range(x.shape[0]):
        v = np.repeat(x[i,:][None, :], s, axis=0)
        hoge.append(v)
    return np.stack(hoge)
