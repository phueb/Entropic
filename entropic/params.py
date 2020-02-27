

param2requests = {
    'period_probability': [(0.1, 0.1)],
    'distractors_after_delay': [True, False]
}


param2default = {
    # rnn
    'hidden_size': 128,  # 8, 32 are too low
    # toy corpus
    'doc_size': 400_000,
    'delay': 200_000,
    'num_xws': 512,
    'num_types': 1024,
    'num_fragments': 4,
    'period_probability': (0.1, 0.0),  # (prob before delay, prob after delay)
    'distractors_after_delay': True,  # False if last category should be trained on exclusively after delay
    # training
    'xws_in_slot_1_only': False,  # when False, 'phantom category' is only visible with period prob > 0
    'slide_size': 64,
    'optimizer': 'adagrad',
    'batch_size': 64,
    'lr': 0.01,
}
