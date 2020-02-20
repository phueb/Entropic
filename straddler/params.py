

param2requests = {
    'num_fragments': [2, 3, 4],
    'optimizer': ['adagrad'],
}


param2default = {
    # rnn
    'hidden_size': 512,
    # toy corpus
    'doc_size': 100_000,
    'num_xws': 512,
    'num_types': 4096,
    'num_fragments': 2,
    'fragmentation_prob': 1.0,
    # training
    'slide_size': 64,
    'optimizer': 'adagrad',
    'batch_size': 64,
    'lr': 0.01,
}
