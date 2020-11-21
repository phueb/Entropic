"""

'redundant_a': ((0.0, 0.1), (0.6, 1.0)) means:
redundancy is gradually ramped from 0.0 to 0.1 in document 1, and then from 0.6 to 1.0 in document 2.

"""

param2requests = {
    # 'flavor': ['lstm'],
    # 'lr': [1.0],
    'redundant_a': [((0.8, 1.0), (1.0, 1.0)),
                        ((0.9, 1.0), (1.0, 1.0)),
                        ((1.0, 1.0), (1.0, 1.0))],

}

param2debug = {
    'doc_size': 1_000,
}

param2default = {
    # rnn
    'hidden_size': 64,
    'flavor': 'srn',
    # toy corpus
    'doc_size': 50_000,
    'num_types': 128,
    'num_fragments': 4,
    'starvation': ((0.0, 0.0), (0.0, 0.0)),
    'redundant_a': ((0.0, 0.0), (0.0, 0.0)),
    'redundant_b': ((0.0, 0.0), (0.0, 0.0)),
    'size_a': ((1.0, 1.0), (1.0, 1.0)),
    'size_b': ((1.0, 1.0), (1.0, 1.0)),
    'drop_a': ((0.0, 0.0), (0.0, 0.0)),
    'drop_b': ((0.0, 0.0), (0.0, 0.0)),
    # training
    'optimizer': 'sgd',
    'lr': 0.4,  # 0.01 for adagrad, 0.5 for sgd
    'batch_size': 64,
}
