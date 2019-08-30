import numpy as np


def make_superordinate_w1(num_hidden):
    a = np.tile(np.random.standard_normal(size=(1, num_hidden)), (9, 1))  # same weights for inputs in cat A
    b = np.tile(np.random.standard_normal(size=(1, num_hidden)), (9, 1))  # same weights for inputs in cat B
    res = np.vstack((a, b))  # [18, num_hidden]
    return res


def make_identical_w1(num_hidden):
    res = np.tile(np.random.standard_normal(size=(1, num_hidden)), (18, 1))  # same weights for inputs in cat A and B
    return res