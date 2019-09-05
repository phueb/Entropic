import numpy as np


def make_superordinate_w1(num_cols, num_input_a, num_input_b):
    a = np.tile(np.random.standard_normal(size=(1, num_cols)), (num_input_a, 1))  # same weights for items in A
    b = np.tile(np.random.standard_normal(size=(1, num_cols)), (num_input_b, 1))  # same weights for items in B
    res = np.vstack((a, b))  # [num_items_in_a + num_items_in_b, num_cols]
    return res


def make_identical_w1(num_cols, num_input_a, num_input_b):
    num_input = num_input_a + num_input_b
    res = np.tile(np.random.standard_normal(size=(1, num_cols)), (num_input, 1))  # same weights for all items
    return res
