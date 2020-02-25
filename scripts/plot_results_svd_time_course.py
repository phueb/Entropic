import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from entropic import config
from entropic.params import param2default, param2requests

from ludwig.results import gen_param_paths


LABEL_PARAMS = []  # must be a list

# collect data
summary_data = []
for param_p, label in gen_param_paths(config.Dirs.root.name,
                                      param2requests,
                                      param2default,
                                      label_params=LABEL_PARAMS):
    # param_df
    dfs = []
    for df_p in param_p.glob(f'*num*/sing-dim*.csv'):
        print('Reading {}'.format(df_p.name))
        df = pd.read_csv(df_p, index_col=0)
        df.index.name = 'epoch'
        dfs.append(df)
    param_df = frame = pd.concat(dfs, axis=1)
    print(param_df)

    # TODO make svd time course plot


# plot

plt.show()
