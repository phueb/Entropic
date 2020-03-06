

param2requests = {
    'period_probability': [(0.0, 0.05), (0.05, 0.00)],
}


param2default = {
    # rnn
    'hidden_size': 128,  # 8, 32 are too low
    # toy corpus
    'doc_size': 400_000,
    'delay': 200_000,
    'num_types': 128,
    'num_fragments': 4,
    'period_probability': (0.0, 0.0),  # (prob before delay, prob after delay)
    'num_sentinels': 4,  # number of examples of each x-word category seen before delay
    # training
    'xws_in_slot_1_only': False,  # when False, 'phantom category' is only visible with period prob > 0
    'slide_size': 64,
    'optimizer': 'adagrad',
    'batch_size': 64,
    'lr': 0.01,
}
