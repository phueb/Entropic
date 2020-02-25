

param2requests = {
    'num_fragments': [2],
    'doc_size': [100_000],
    'period_probability': [0.1, 0.0],
    'xws_in_slot_1_only': [True],
}


param2default = {
    # rnn
    'hidden_size': 128,  # TODO
    # toy corpus
    'doc_size': 500_000,
    'num_xws': 512,
    'num_types': 1024,
    'num_fragments': 2,
    'period_probability': 0.0,
    # training
    'xws_in_slot_1_only': True,  # True results in "phantom categories"
    'slide_size': 64,
    'optimizer': 'adagrad',
    'batch_size': 64,
    'lr': 0.01,
}
