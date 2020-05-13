import numpy as np


def onehot(data, n):
    """onehot ecoder"""
    buf = np.zeros(data.shape + (n,))
    nmsk = np.arange(data.size) * n + data.ravel()
    buf.ravel()[nmsk - 1] = 1
    return buf
