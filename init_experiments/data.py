import numpy as np
import torch


class Data:
    """
    structure of the data:
    there are 2 superordinate categories (category A and B).
    each superordinate category contains NUM_SUBORDINATES subordinate categories.
    each subordinate category contains SUBORDINATE_SIZE items.

    y1 has subordinate category feedback
    y2 has superordinate category feedback
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

        # y2_random results in worst performance (no similarity between members of same subordinate
        # y2_subordinates_identical results in best performance (max similarity between members of same subordinate)
        # do not simply create a random y2 matrix because members of same subordinate are exposed to same statistics
        # which makes representations between members of same subordinate more similar
        # instead, flip between two feedback modes
        # this simulates that a word often appears in the correct superordinate (noun) context and
        # often appears in one different superordinate (e.g. verb or adjective) context

        if epoch < self.params.y2_gold_on[0]:
            gold_prob = self.params.y2_gold_on[1]
        else:
            gold_prob = self.params.y2_gold_on[2]

        y2 = self.chose_y2_rows(gold_prob)

        res = self.y1_gold + y2
        return res.astype(np.float32)

    def chose_y2_rows(self, gold_prob):
        """
        return matrix by choosing randomly rows from y2_random and y2_gold
        """
        rand_ints = np.random.choice([1, 0],
                                     p=[gold_prob, 1 - gold_prob],
                                     size=self.input_size)
        res = np.matmul(np.diag(rand_ints - 0), self.y2_gold) + \
              np.matmul(np.diag(1 - rand_ints), self.y2_random)  # gold has to be first
        return res
