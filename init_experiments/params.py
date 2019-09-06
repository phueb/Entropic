

class ObjectView(object):
    def __init__(self, d: object):
        self.__dict__ = d


class Params:
    init = ['identical', 'linear', 'superordinate', 'random']
    scale_weights = [1.0]  # works with 1.0 but not with 0.01 or 0.1
    lr = [1.0]
    hidden_size = [8]
    num_epochs = [5 * 1000]

    separate_feedback = [[True, 0.5]]  # probability of using only subordinate feedback for a single item
    y2_noise = [[False, 0.1]]  # probability of switching the superordinate label for a single item

    num_evals = [10]
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

    separate_feedback = [[True, 0.5]]  # probability of using only subordinate feedback for a single item
    y2_noise = [[False, 0.1]]  # probability of switching the superordinate label for a single item

    num_evals = [10]
    representation = ['output']

    num_subordinate_cats_in_a = [3]
    num_subordinate_cats_in_b = [3]
    subordinate_size = [3]