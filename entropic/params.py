

param2requests = {
    'period_probability': [0.0, 0.1],
    'num_fragments': [4],
    'xws_in_slot_1_only': [False],
}


param2default = {
    # rnn
    'hidden_size': 128,  # TODO how low?
    # toy corpus
    'doc_size': 100_000,
    'num_xws': 512,
    'num_types': 1024,
    'num_fragments': 2,
    'period_probability': 0.0,
    # training
    'xws_in_slot_1_only': False,  # when False, 'phantom category' is only visible with period prob > 0
    'slide_size': 64,
    'optimizer': 'adagrad',
    'batch_size': 64,
    'lr': 0.01,
}
