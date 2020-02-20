

param2requests = {
    'num_fragments': [1, 2, 3, 4],
}


param2default = {
    # rnn
    'batch_size': 64,
    'lr': 0.01,
    'hidden_size': 512,
    # toy corpus
    'doc_size': 100_000,
    'num_xws': 512,
    'num_types': 4096,
    'num_fragments': 2,
    'fragmentation_prob': 1.0,
    # training
    'slide_size': 64,
}
