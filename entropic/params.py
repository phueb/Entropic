"""
WARNING:
to reduce confound of presence of periods reducing signal from y compared to not having periods,
it is best to compare (0.05, 0.5) vs (0.0, 0.05) such that periods occur equally frequent in both conditions
after the delay.
equivalently: compare (0.05, 0.0) vs (0.0, 0.0)

Notes:
all words occur approximately equally in  corpus, except sentinels,
which occur more often than non-sentinels

starvation: (see gradient starvation in  Combes et al., 2018)
the probability that Yi is a semantically uninformative symbol (kind of like a period)
"""

param2requests = {
    # 'sample_a': [('super', 'super'), ('sub', 'sub'), ('item', 'item')] +
    #             [('super', 'item'), ('item', 'super')],
    # 'sample_b': [('super', 'super'), ('sub', 'sub'), ('item', 'item')] +
    #             [('super', 'item'), ('item', 'super')],


    'sample_a': [('item', 'item')],
    'incongruent_a': [(0.0, 0.0), (0.1, 0.1), (0.2, 0.2), (0.3, 0.3), (0.4, 0.4), (0.5, 0.5),
                      (0.6, 0.6), (0.7, 0.7), (0.8, 0.8), (0.9, 0.9), (1.0, 1.0)],

    # 'incongruent_b': [(0.0, 0.0), (0.1, 0.1), (0.2, 0.2), (0.3, 0.3), (0.4, 0.4), (0.5, 0.5),
    #                   (0.6, 0.6), (0.7, 0.7), (0.8, 0.8), (0.9, 0.9), (1.0, 1.0)],

}

param2debug = {
    'size_a': (1.1, 1.0),
    'num_sentinels': 8,
}

param2default = {
    # rnn
    'hidden_size': 64,  # 8, 32 are too low
    # toy corpus
    'doc_size': 100_000,
    'delay': 50_000,
    'num_types': 128,
    'num_fragments': 4,
    'starvation': (0.0, 0.0),  # (prob before delay, prob after delay)
    'num_sentinels': 8,  # number of examples of each x-word category seen before delay
    'sample_a': ('super', 'super'),
    'sample_b': ('super', 'super'),
    'incongruent_a': (0.0, 0.0),  # probability that Ai is category incongruent
    'incongruent_b': (0.0, 0.0),
    'size_a': (1.0, 1.0),  # proportion of set size of A
    'size_b': (1.0, 1.0),
    # training
    'optimizer': 'sgd',
    'lr': 0.4,  # 0.01 for adagrad, 0.5 for sgd
    'batch_size': 64,
}
