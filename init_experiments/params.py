

class DefaultParams:

    def __init__(self):
        self.init = 'random'
        self.scale_weights = 1.0  # works with 1.0 but not with 0.01 or 0.1
        self.lr = 1.0
        self.hidden_size = 8
        self.num_epochs = 5 * 1000

        self.y2_feedback = True
        self.separate_feedback = [0, 0.0]  # P of using only subordinate feedback for a single item
        self.y2_noise = [0, 0.0]  # P of switching the superordinate label for a single item

        self.representation = 'output'

        self.num_subordinate_cats_in_a = 3
        self.num_subordinate_cats_in_b = 3
        self.subordinate_size = 3

        self.param_name = 'default'
        self.job_name = 'default'

    def __str__(self):
        res = ''
        for k, v in sorted(self.__dict__.items()):
            res += '{}={}\n'.format(k, v)
        return res

    def __eq__(self, other):
        print('Checking equivalence of two params:')
        for k in self.__dict__:
            if k not in other.__dict__:
                return False
            if other.__dict__[k] != self.__dict__[k]:
                print('Not equivalent')
                return False
        else:
            print('Equivalent')
            return True


class Params(DefaultParams):

    def __init__(self, param2val):
        super().__init__()

        for k, v in param2val.copy().items():
            if k not in self.__dict__:
                raise KeyError('Error updating default params: "{}" not in default params.'.format(k))
            self.__dict__[k] = v


# specify params to submit here
partial_request = {'y2_noise': [[0, 0.5], [100, 0.5], [1000, 0.5]]}