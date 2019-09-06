

class ObjectView(object):
    def __init__(self, d: object):
        self.__dict__ = d


class Params:
    init = ['identical', 'superordinate', 'linear', 'random']
    scale_weights = [1.0]  # works with 1.0 but not with 0.01 or 0.1
    lr = [1.0]
    hidden_size = [8]
    num_epochs = [5 * 1000]

    y2_feedback = [True]
    separate_feedback = [[False, 0.0]]  # P of using only subordinate feedback for a single item
    y2_noise = [[False, 0.0]]  # P of switching the superordinate label for a single item

    representation = ['output']

    num_subordinate_cats_in_a = [3]
    num_subordinate_cats_in_b = [3]
    subordinate_size = [3]


class DefaultParams:
    init = ['identical']
    scale_weights = [1.0]  # works with 1.0 but not with 0.01 or 0.1
    lr = [1.0]
    hidden_size = [8]
    num_epochs = [5 * 1000]

    y2_feedback = [True]  # TODO test
    separate_feedback = [[False, 0.0]]  # probability of using only subordinate feedback for a single item
    y2_noise = [[False, 0.0]]  # probability of switching the superordinate label for a single item

    representation = ['output']

    num_subordinate_cats_in_a = [3]
    num_subordinate_cats_in_b = [3]
    subordinate_size = [3]