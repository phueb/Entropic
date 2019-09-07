"""
y2_flip doesn't result in worse learning because it affects members of a superordinate consistently on average.
each member has the same probability that superordinate feedback is switched from category A to B and vice versa.
the point is that each member is equally affected and this similarity should push them closer together
in representational space
"""

"""
use only dictionaries to store parameters.
ludwigcluster works on dictionaries and any custom class would force potentially unwanted logic on user.
using non-standard classes here would also make it harder for user to understand.
any custom classes for parameters should be implemented by user in main job function only.
keep interface between user and ludwigcluster as simple as possible
"""


param2default = {
    'init': 'random',
    'scale_weights': 1.0,  # works with 1.0 but not with 0.01 or 0.1
    'lr': 1.0,
    'hidden_size': 8,
    'num_epochs': 5 * 1000,
    'y2_feedback': True,
    'separate_feedback': [0, 0.0],  # P of using only subordinate feedback for a single item
    'y2_flip': [0, 0.0],  # P of switching the superordinate label for a single item
    'y2_static_noise': 0,  # epoch before which to apply static noise to y2
    'representation': 'output',
    'num_subordinate_cats_in_a': 3,
    'num_subordinate_cats_in_b': 3,
    'subordinate_size': 3
}


# specify params to submit here
param2requests = {'y2_static_noise': [100, 0],
                  'init': ['random', 'identical']}