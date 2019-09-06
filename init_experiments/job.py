import numpy as np
import torch
import sys
from sklearn.metrics.pairwise import cosine_similarity
from torch import optim as optim
import yaml
import pandas as pd

from init_experiments.data import Data
from init_experiments.params import Params
from init_experiments import config
from init_experiments.eval import calc_cluster_score
from init_experiments.net import Net
from init_experiments.utils import to_eval_epochs


def main(param2val):

    # params
    params = Params(param2val)
    print(params)
    sys.stdout.flush()

    # data
    data = Data(params)

    # net
    net = Net(params, data)

    # optimizer + criterion
    optimizer = optim.SGD(net.parameters(), lr=params.lr)
    criterion = torch.nn.MSELoss()

    # eval before start of training
    net.eval()
    torch_o = net(data.torch_x)
    torch_y = torch.from_numpy(data.y)
    loss = criterion(torch_o, torch_y)
    net.train()

    # train loop
    eval_epoch_idx = 0
    scores_a = np.zeros(config.Eval.num_evals)
    scores_b = np.zeros(config.Eval.num_evals)
    eval_epochs = to_eval_epochs(params)
    for epoch in range(params.num_epochs + 1):
        # eval
        if epoch in eval_epochs:
            # cluster score
            print('Evaluating at epoch {}'.format(epoch))
            collect_scores(data, params, net, eval_epoch_idx, scores_a, scores_b, torch_o)
            eval_epoch_idx += 1
            # mse
            mse = loss.detach().numpy().item()
            print('mse={}'.format(mse))
            print()
            sys.stdout.flush()

        y = adjust_y(data, params, epoch).astype(np.float32)

        # train
        optimizer.zero_grad()  # zero the gradient buffers
        torch_o = net(data.torch_x)  # feed-forward
        torch_y = torch.from_numpy(y)
        loss = criterion(torch_o, torch_y)  # compute loss
        loss.backward()  # back-propagate
        optimizer.step()  # update

    # to pandas
    eval_epochs = to_eval_epochs(params)
    df_a = pd.DataFrame(scores_a, index=eval_epochs, columns=[config.Eval.metric])
    df_b = pd.DataFrame(scores_b, index=eval_epochs, columns=[config.Eval.metric])

    if config.Eval.debug:
        print(df_a)
        print(df_b)
        raise SystemExit('Debugging: Not saving results')

    # save data - do not create directory on shared drive until all results are available
    dst = config.RemoteDirs.runs / param2val['param_name'] / param2val['job_name']
    if not dst.exists():
        dst.mkdir(parents=True)
    with (dst / 'results.npy').open('wb') as f:
        np.savez(f, {'scores_a': scores_a, 'scores_b': scores_b})
    with (dst / 'results_a.csv').open('w') as f:
        df_a.to_csv(f, index=True)
    with (dst / 'results_b.csv').open('w') as f:
        df_b.to_csv(f, index=True)

    # write param2val to shared drive
    param2val_p = config.RemoteDirs.runs / param2val['param_name'] / 'param2val.yaml'
    if not param2val_p.exists():
        param2val['job_name'] = None
        with param2val_p.open('w', encoding='utf8') as f:
            yaml.dump(param2val, f, default_flow_style=False, allow_unicode=True)


def adjust_y(data, params, epoch):
    # shuffle superordinate feedback
    if epoch < params.y2_noise[0]:
        is_noise_list = np.random.choice([True, False],
                                         p=[params.y2_noise[1], 1 - params.y2_noise[1]],
                                         size=data.input_size)
        indices = np.array([[1, 0] if is_noise else [0, 1] for is_noise in is_noise_list])
        sup_cols = np.take_along_axis(data.sup_cols_template, indices, axis=1)
        y2 = np.hstack((np.zeros((data.sub_cols.shape[0], data.sub_cols.shape[1])), sup_cols))
        y = data.y1 + y2

        print(epoch)
        print('adjusting y')  # TODO test
        print(y)

    else:
        y2 = data.y2.copy()
        y = data.y.copy()
    # give feedback from either y1 or y2 but never together
    if epoch < params.separate_feedback[0]:
        row_ids = [idx1 if np.random.binomial(n=1, p=params.separate_feedback[1]) else idx2
                   for idx1, idx2 in zip(data.y1_ids, data.y2_ids)]
        y = np.vstack((data.y1, y2))[row_ids].astype(np.float32)  # select rows from y1 OR y2
    return y


def collect_scores(data, params, net, eval_epoch_idx, scores_a, scores_b, torch_o):
    # rep_mat_a_b
    if params.representation == 'output':
        rep_mat_a_b = torch_o.detach().numpy()  # TODO make gif of evolving prediction mat
    elif params.representation == 'hidden':
        rep_mat_a_b = net.linear1(data.torch_x).detach().numpy()
    else:
        raise AttributeError('Invalid arg to "representation".')
    #
    rep_mats = [rep_mat_a_b[:data.input_size_a], rep_mat_a_b[-data.input_size_b:]]
    rep_mats_gold = [data.y[:data.input_size_a], data.y[-data.input_size_b:]]
    for scores, rep_mat, rep_mat_gold in zip([scores_a, scores_b], rep_mats, rep_mats_gold):
        if scores[max(0, eval_epoch_idx - 1)] == 1.0:
            continue
        sim_mat = cosine_similarity(rep_mat)
        sim_mat_gold = np.rint(cosine_similarity(rep_mat_gold))
        score = calc_cluster_score(sim_mat, sim_mat_gold, config.Eval.metric)
        scores[eval_epoch_idx] = score
        #
        if score == 1.0:
            scores[eval_epoch_idx:] = 1.0
        #
        print(rep_mat[:data.input_size_a].round(3))
        print(sim_mat_gold.round(1))
        print(sim_mat.round(4))
        print('{}_a={}'.format(config.Eval.metric, score)) if config.Eval.score_a else None
        print()