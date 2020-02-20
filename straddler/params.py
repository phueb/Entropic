

param2requests = {'y2_gold_on': [[2500, 0.9, 0.5], [2500, 0.5, 0.9]],
                  'init': ['random']}


param2default = {
    'init': 'random',
    'scale_weights': 1.0,  # works with 1.0 but not with 0.01 or 0.1
    'lr': 1.0,
    'hidden_size': 8,
    'num_epochs': 5 * 1000,
    'y2_gold_on': [0, 0.0, 0.0],  # [epoch, P(y2 feedback before epoch), P(2 feedback after epoch)
    'representation': 'output',
    'num_subordinate_cats': 3,
    'subordinate_size': 3
}
