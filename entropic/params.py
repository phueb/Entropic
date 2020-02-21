

param2requests = {
    'num_fragments': [2],
    'doc_size': [100_000],
    'fragmentation_prob': [1.0],  # TODO
}


param2default = {
    # rnn
    'hidden_size': 128,  # TODO
    # toy corpus
    'doc_size': 5_000_000,
    'num_xws': 512,
    'num_types': 1024,
    'num_fragments': 2,
    'fragmentation_prob': 1.0,
    # training
    'slide_size': 64,
    'optimizer': 'adagrad',
    'batch_size': 64,
    'lr': 0.01,
}
