import numpy as np
import torch


class Data:
    """
    structure of the self:
    there are 2 superordinate categories (category A and B).
    each superordinate category contains NUM_SUBORDINATES subordinate categories.
    each subordinate category contains SUBORDINATE_SIZE items.
    """

    def __init__(self, params):
        self.params = params
        self.superordinate_size = self.params.subordinate_size * self.params.num_subordinate_cats
        self.input_size = 2 * self.superordinate_size
        self.output_size = 2 * 2 * self.params.num_subordinate_cats  # a and b, y1 and y2 each have num_subordinate_cats

        # x
        self.x = self.make_x()
        self.torch_x = torch.from_numpy(self.x)

        # y
        self.sub_cols_gold = self.make_sub_cols_gold()  # subordinate category feedback
        self.sup_cols_gold = self.make_sup_cols_gold()  # superordinate category feedback
        self.y1_gold = np.hstack((self.sub_cols_gold, np.zeros_like(self.sup_cols_gold)))
        self.y2_gold = np.hstack((np.zeros_like(self.sub_cols_gold), self.sup_cols_gold))
        assert self.y1_gold.shape[0] == self.input_size
        assert self.y1_gold.shape[1] == self.output_size
        assert self.y2_gold.shape[0] == self.input_size
        assert self.y2_gold.shape[1] == self.output_size

        # improves learning because each subordinate is made more similar
        self.y2_subordinates_identical = np.roll(self.y1_gold, self.output_size // 2)

        self.y2_random = np.random.permutation(self.y2_subordinates_identical)

        assert np.sum(self.y1_gold + self.y2_gold) == 2 * self.input_size  # each item has sub and super ordinate cat

    def make_x(self):
        x = np.eye(self.input_size, dtype=np.float32)
        return x

    def make_sub_cols_gold(self):
        res = np.repeat(np.eye(2 * self.params.num_subordinate_cats), self.params.subordinate_size, axis=0)

        assert res.shape[0] == self.input_size
        assert res.shape[1] == self.output_size // 2

        return res.astype(np.float32)

    def make_sup_cols_gold(self):

        res = np.zeros((self.input_size, 2 * self.params.num_subordinate_cats))
        res[:, 0] = np.array([[1], [0]]).repeat(self.input_size // 2)
        res[:, 1] = np.array([[0], [1]]).repeat(self.input_size // 2)

        assert res.shape[0] == self.input_size
        assert res.shape[1] == self.output_size // 2

        return res.astype(np.float32)

    def make_y(self, epoch):


        # TODO test

        # at each update step, chose randomly between two y2 feedback options:
        # y2_random results in worst performance (no similarity between members of same subordinate
        # y2_subordinates_identical results in best performance (max similarity between members of same subordinate)
        # do not simply create a random y2 matrix because members of same subordinate are exposed to same statistics
        # which makes representations between members of same subordinate more similar

        # this creates y2 by choosing randomly a row from either y2_random or y2_subordinates_identical
        rand_ints = np.random.random_integers(0, 1, size=self.input_size)
        y2 = np.matmul(np.diag(rand_ints - 0), self.y2_random) + \
             np.matmul(np.diag(1 - rand_ints), self.y2_subordinates_identical)

        if epoch < self.params.y2_gold_on[0]:  # if y2 is turned on, gold (superordinate) feedback is given
            if np.random.binomial(n=1, p=self.params.y2_gold_on[1]):
                y2 = self.y2_gold.copy()
        else:
            if np.random.binomial(n=1, p=self.params.y2_gold_on[2]):
                y2 = self.y2_gold.copy()

        res = self.y1_gold + y2  # y1 has subordinate category feedback, y2 has superordinate category feedback
        return res.astype(np.float32)

