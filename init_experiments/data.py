import numpy as np
import torch


class Data:
    """
    structure of the data:
    there are 2 superordinate categories (category A and B).
    each superordinate category contains NUM_SUBORDINATES subordinate categories.
    each subordinate category contains SUBORDINATE_SIZE items.
    """

    def __init__(self, params):
        self.params = params
        self.input_size_a = self.params.subordinate_size * self.params.num_subordinate_cats_in_a
        self.input_size_b = self.params.subordinate_size * self.params.num_subordinate_cats_in_b
        self.input_size = self.input_size_a + self.input_size_b
        # for probabilistic supervision - sample feedback from either y1 or y2
        self.y1_ids = np.arange(self.input_size)
        self.y2_ids = self.y1_ids + self.input_size
        #
        self.x = self.make_x()
        self.torch_x = torch.from_numpy(self.x)
        #
        self.sub_cols_template = self.make_sub_cols().astype(np.float32)  # subordinate category feedback
        self.sup_cols_template = self.make_sup_cols().astype(np.float32)  # superordinate category feedback
        self.sub_cols = self.sub_cols_template.copy().astype(np.float32)
        self.sup_cols = self.sup_cols_template.copy().astype(np.float32)
        self.y1 = np.hstack((self.sub_cols,
                             np.zeros((self.sup_cols.shape[0], self.sup_cols.shape[1])))).astype(np.float32)
        self.y2 = np.hstack((np.zeros((self.sub_cols.shape[0], self.sub_cols.shape[1])),
                             self.sup_cols)).astype(np.float32)
        self.y = self.y1 + self.y2  # y1 has subordinate category feedback, y2 has superordinate category feedback
        self.output_size = self.y.shape[1]

    def make_x(self):
        x = np.eye(self.input_size, dtype=np.float32)
        return x

    def make_sub_cols(self):
        a_sub = np.hstack((np.eye(self.params.num_subordinate_cats_in_a).repeat(self.params.subordinate_size, axis=0),
                           np.zeros((self.input_size_a, self.params.num_subordinate_cats_in_a))))
        b_sub = np.hstack((np.zeros((self.input_size_b, self.params.num_subordinate_cats_in_b)),
                           np.eye(self.params.num_subordinate_cats_in_b).repeat(self.params.subordinate_size, axis=0)))
        return np.vstack((a_sub, b_sub))

    def make_sup_cols(self):
        a_sup = np.array([[1, 0] if self.params.y2_feedback else [0, 0]]).repeat(self.input_size_a, axis=0)
        b_sup = np.array([[0, 1] if self.params.y2_feedback else [0, 0]]).repeat(self.input_size_b, axis=0)
        return np.vstack((a_sup, b_sup))