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
    'sample_a': [('super', 'super'), ('sub', 'sub'), ('item', 'item')] +
                [('super', 'item'), ('item', 'super')],
    'sample_b': [('super', 'super'), ('sub', 'sub'), ('item', 'item')] +
                [('super', 'item'), ('item', 'super')],

}

param2debug = {
    'sample_a': 'super',
    'sample_b': 'super',
}

param2default = {
    # rnn
    'hidden_size': 64,  # 8, 32 are too low
    # toy corpus
    'doc_size': 200_000,
    'delay': 50_000,
    'num_types': 128,
    'num_fragments': 4,
    'starvation': (0.0, 0.0),  # (prob before delay, prob after delay)
    'num_sentinels': 4,  # number of examples of each x-word category seen before delay
    'sample_a': ('super', 'super'),
    'sample_b': ('super', 'super'),
    # training
    'slide_size': 64,
    'optimizer': 'adagrad',
    'batch_size': 64,
    'lr': 0.01,
}
