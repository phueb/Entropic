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
        self.input_size = 2 * self.params.subordinate_size * self.params.num_subordinate_cats
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

        self.y_gold = self.y1_gold + self.y2_gold
        assert np.sum(self.y_gold) == 2 * self.input_size  # each item is associated with sub and superordinate label

        # use with params.y2_static_noise
        self.rand_probs = np.random.rand(self.input_size)

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
        print('Making y at epoch={}'.format(epoch))

        y2 = self.y2_gold.copy()

        if epoch < self.params.y2_static_noise:
            indices = np.array([[1, 0] if np.random.binomial(n=1, p=p) else [0, 1]
                                for p in self.rand_probs])
            sup_cols = np.take_along_axis(self.sup_cols_gold, indices, axis=1)
            y2 = np.hstack((np.zeros_like(self.sub_cols_gold), sup_cols))

        if epoch < self.params.y2_feedback[0]:  # if no y2 feedback specified, provide uninformative y2 feedback
            if not np.random.binomial(n=1, p=self.params.y2_feedback[1]):
                y2 = np.roll(self.y1_gold, self.output_size // 2)
        else:
            if not np.random.binomial(n=1, p=self.params.y2_feedback[2]):
                y2 = np.roll(self.y1_gold, self.output_size // 2)

        res = self.y1_gold + y2  # y1 has subordinate category feedback, y2 has superordinate category feedback
        return res.astype(np.float32)

