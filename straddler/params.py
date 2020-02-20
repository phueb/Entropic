

param2requests = {
    'num_fragments': [8],
    'optimizer': ['adagrad'],
    'doc_size': [1_000_000],
}


param2default = {
    # rnn
    'hidden_size': 128,  # TODO
    # toy corpus
    'doc_size': 5_000_000,
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
