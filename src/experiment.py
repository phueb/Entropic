import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch import optim as optim

from src import config
from src.eval import calc_cluster_score
from src.net import Net
from src.utils import make_superordinate_w1, make_identical_w1, to_eval_epochs


def simulate(init, opts):
    print('------------------------------------------')
    print('init={}'.format(init))
    print('------------------------------------------')

    # x
    input_size_a = opts.subordinate_size * opts.num_subordinate_cats_in_a
    input_size_b = opts.subordinate_size * opts.num_subordinate_cats_in_b
    input_size = input_size_a + input_size_b
    x = np.eye(input_size, dtype=np.float32)
    torch_x = torch.from_numpy(x.astype(np.float32))
    # y
    a_sub = np.hstack((np.eye(opts.num_subordinate_cats_in_a).repeat(opts.subordinate_size, axis=0),
                       np.zeros((input_size_a, opts.num_subordinate_cats_in_a))))
    b_sub = np.hstack((np.zeros((input_size_b, opts.num_subordinate_cats_in_b)),
                       np.eye(opts.num_subordinate_cats_in_b).repeat(opts.subordinate_size, axis=0)))
    a_sup = np.array([[1, 0]]).repeat(input_size_a, axis=0)
    b_sup = np.array([[0, 1]]).repeat(input_size_b, axis=0)
    sub_cols = np.vstack((a_sub, b_sub))
    sup_cols = np.vstack((a_sup, b_sup))
    sup_cols_template = sup_cols.copy()  # may be modified in training loop
    y1 = np.hstack((sub_cols, np.zeros((sup_cols.shape[0], sup_cols.shape[1]))))
    y2 = np.hstack((np.zeros((sub_cols.shape[0], sub_cols.shape[1])), sup_cols))
    y = y1 + y2  # y1 has subordinate category feedback, y2 has superordinate category feedback

    # gold labels for category structure in A and B
    gold_sim_mat_a = np.rint(cosine_similarity(y[:input_size_a]))
    gold_sim_mat_b = np.rint(cosine_similarity(y[-input_size_b:]))

    # for probabilistic supervision - sample feedback from either y1 or y2
    y1_ids = np.arange(y1.shape[0])
    y2_ids = y1_ids + y1.shape[0]

    # net
    output_size = y.shape[1]
    if init == 'random':
        w1 = np.random.standard_normal(size=(input_size, opts.hidden_size)) * opts.scale_weights
    elif init == 'superordinate':
        w1 = make_superordinate_w1(opts.hidden_size, input_size_a, input_size_b) * opts.scale_weights
    elif init == 'identical':
        w1 = make_identical_w1(opts.hidden_size, input_size_a, input_size_b) * opts.scale_weights
    elif init == 'linear':  # each item is assigned same weight vector with linear transformation
        w1 = make_identical_w1(opts.hidden_size, input_size_a, input_size_b) * opts.scale_weights
        w1_delta = np.tile(np.linspace(0, 1, num=w1.shape[0])[:, np.newaxis], (1, w1.shape[1]))
        w1 += np.random.permutation(w1_delta)  # permute otherwise linear transform will align with category structure
    else:
        raise AttributeError('Invalid arg to "init".')
    net = Net(input_size=input_size,
              hidden_size=opts.hidden_size,
              output_size=output_size,
              w1=w1,
              w2=np.random.standard_normal(size=(opts.hidden_size, output_size)) * opts.scale_weights)

    # optimizer + criterion
    optimizer = optim.SGD(net.parameters(), lr=opts.lr)
    criterion = torch.nn.MSELoss()

    # eval before start of training
    net.eval()
    torch_o = net(torch_x)
    net.train()

    # train loop
    eval_score_a = config.Eval.score_a  # save time - don't evaluate ba after it has reached 1.0
    eval_score_b = config.Eval.score_b
    eval_epoch_idx = 0
    score_trajectory_a = np.ones(opts.num_evals)
    score_trajectory_b = np.ones(opts.num_evals)
    eval_epochs = to_eval_epochs(opts)
    for epoch in range(opts.num_epochs + 1):
        # eval
        if epoch in eval_epochs:
            print('Evaluating at epoch {}'.format(epoch))
            if opts.representation == 'output':
                prediction_mat = torch_o.detach().numpy()  # TODO make gif of evolving prediction mat
            elif opts.representation == 'hidden':
                prediction_mat = net.linear1(torch_x).detach().numpy()
            else:
                raise AttributeError('Invalid arg to "REPRESENTATION".')
            sim_mat_a = cosine_similarity(prediction_mat[:input_size_a])
            sim_mat_b = cosine_similarity(prediction_mat[-input_size_b:])
            score_a = calc_cluster_score(sim_mat_a, gold_sim_mat_a, config.Eval.metric) if eval_score_a else 1.0
            score_b = calc_cluster_score(sim_mat_b, gold_sim_mat_b, config.Eval.metric) if eval_score_b else 1.0
            score_trajectory_a[eval_epoch_idx] = score_a
            score_trajectory_b[eval_epoch_idx] = score_b
            eval_epoch_idx += 1
            #
            print(prediction_mat[:input_size_a].round(3))
            print(prediction_mat[-input_size_b:].round(3))
            print(gold_sim_mat_a.round(1))
            print(sim_mat_a.round(4))
            print('{}_a={}'.format(config.Eval.metric, score_a)) if config.Eval.score_a else None
            print('{}_b={}'.format(config.Eval.metric, score_b)) if config.Eval.score_b else None
            print()
            #
            if score_a == 1.0:
                eval_score_a = False
            if score_b == 1.0:
                eval_score_b = False

        # shuffle superordinate feedback   # TODO make this a function of epoch id
        if opts.y2_noise[0]:
            is_noise_list = np.random.choice([True, False],
                                             p=[opts.y2_noise[1], 1 - opts.y2_noise[1]],
                                             size=y1.shape[0])
            indices = np.array([[1, 0] if is_noise else [0, 1] for is_noise in is_noise_list])
            sup_cols = np.take_along_axis(sup_cols_template, indices, axis=1)
            y2 = np.hstack((np.zeros((sub_cols.shape[0], sub_cols.shape[1])), sup_cols))
            y = y1 + y2

        # give feedback from either y1 or y2 but never together
        if opts.separate_feedback[0]:
            row_ids = [idx1 if np.random.binomial(n=1, p=opts.separate_feedback[1]) else idx2
                       for idx1, idx2 in zip(y1_ids, y2_ids)]
            y = np.vstack((y1, y2))[row_ids]  # select rows from y1 or y2

        # train
        optimizer.zero_grad()  # zero the gradient buffers
        torch_o = net(torch_x)  # feed-forward
        torch_y = torch.from_numpy(y.astype(np.float32))
        loss = criterion(torch_o, torch_y)  # compute loss
        loss.backward()  # back-propagate
        optimizer.step()  # update

        # mse
        if epoch in eval_epochs:
            mse = loss.detach().numpy().item()
            print('mse={}'.format(mse))

    return score_trajectory_a, score_trajectory_b