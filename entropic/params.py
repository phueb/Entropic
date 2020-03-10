"""
WARNING:
to reduce confound of presence of periods reducing signal from y compared to not having periods,
it is best to compare (0.05, 0.5) vs (0.0, 0.05) such that periods occur equally frequent in both conditions
after the delay.
equivalently: compare (0.05, 0.0) vs (0.0, 0.0)

Notes:
all words occur approximately equally in  corpus, except sentinels,
which occur more often than non-sentinels
"""

param2requests = {
    # 'period_probability': [(0.05, 0.00), (0.00, 0.05), (0.00, 0.00), (0.05, 0.05)],
    # 'sample_v': ['target-category', 'superordinate', 'item'],
    # 'sample_w': ['target-category', 'superordinate', 'item'],
    'sample_v': ['item'],
    'sample_w': ['target-category', 'superordinate'],

}


param2default = {
    # rnn
    'hidden_size': 64,  # 8, 32 are too low
    # toy corpus
    'doc_size': 400_000,
    'delay': 200_000,
    'num_types': 128,
    'num_fragments': 4,
    'period_probability': (0.0, 0.0),  # (prob before delay, prob after delay)
    'num_sentinels': 4,  # number of examples of each x-word category seen before delay
    'sample_w': 'target-category',
    'sample_v': 'target-category',
    # training
    'xws_in_slot_1_only': False,  # when False, 'phantom category' is only visible with period prob > 0
    'slide_size': 64,
    'optimizer': 'adagrad',
    'batch_size': 64,
    'lr': 0.01,
}
