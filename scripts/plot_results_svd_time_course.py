import numpy as np

from entropic import config
from entropic.params import param2default, param2requests
from entropic.figs import make_svd_across_time_fig

from ludwig.results import gen_param_paths


LABEL_PARAMS = ['period_probability']  # must be a list

MAX_FILES = 60


# collect data
summary_data = []
for param_p, label in gen_param_paths(config.Dirs.root.name,
                                      param2requests,
                                      param2default,
                                      label_params=LABEL_PARAMS):
    #
    tmp = []
    step = 0
    for npy_path in list(sorted(param_p.glob(f'*num*/saves/output_probabilities*.npy')))[:MAX_FILES]:
        print('Reading {}'.format(npy_path.name))
        representations_all = np.load(npy_path)
        representations_cat1 = representations_all[0::2].mean(0)
        representations_cat2 = representations_all[1::2].mean(0)
        representations_categories = np.vstack((representations_cat1, representations_cat2))
        tmp.append(representations_categories)

        step = int(''.join([i for i in npy_path.name if i.isdigit()]))

    representations = np.stack(tmp)
    assert np.ndim(representations) == 3  # (ticks, words/categories, embedding dimensions)

    # plot
    fig = make_svd_across_time_fig(representations,
                                   words=[f'c-{i+1}' for i in range(representations.shape[1])],
                                   component1=0,
                                   component2=1,
                                   label=label + f'\nlast step={step}',
                                   num_ticks=len(representations))
    fig.show()
