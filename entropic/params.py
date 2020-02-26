

param2requests = {
    'period_probability': [0.0, 0.1],
    'delay': [0, 50_000],
}


param2default = {
    # rnn
    'hidden_size': 128,  # 8, 32 are too low
    # toy corpus
    'doc_size': 200_000,
    'delay': 50_000,
    'num_xws': 512,
    'num_types': 1024,
    'num_fragments': 4,
    'period_probability': 0.0,
    # training
    'xws_in_slot_1_only': False,  # when False, 'phantom category' is only visible with period prob > 0
    'slide_size': 64,
    'optimizer': 'adagrad',
    'batch_size': 64,
    'lr': 0.01,
}
