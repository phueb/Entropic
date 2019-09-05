import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import t

from src.net import Net
from src.eval import calc_cluster_score
from src.plot import plot_trajectories
from src.utils import make_superordinate_w1, make_identical_w1
import torch.optim as optim

VERBOSE = False
CLUSTER_METRIC = 'ba'
EVAL_SCORE_A = True
EVAL_SCORE_B = False  # False to save time


class Opt:

    scale_weights = 1.0  # works with 1.0 but not with 0.01 or 0.1
    lr = 1.0  # TODO vary
    hidden_size = 8
    num_epochs = 1000

    separate_feedback = [True, 0.5]  # probability of sampling from y1 relative to y2
    y2_flip_prob = 0.0  # probability of flipping superordinate feedback
    y2_shuffle_prob = 0.25  # probability of shuffling superordinate feedback (breaks symmetry)

    num_reps = 5
    eval_interval = 100
    representation = 'output'

    num_subordinate_cats_in_a = 3
    num_subordinate_cats_in_b = 3
    subordinate_size = 3


"""
structure of the data:
there are 2 superordinate categories (category A and B).
each superordinate category contains NUM_SUBORDINATES subordinate categories.
each subordinate category contains SUBORDINATE_SIZE items.
"""

eval_epochs = np.arange(0, Opt.num_epochs + 1, Opt.eval_interval)  # +1 because evaluation should happen at last epoch
num_eval_epochs = len(eval_epochs)

if not EVAL_SCORE_A and not EVAL_SCORE_B:
    raise SystemExit('EVAL_SCORE_A and EVAL_SCORE_B are set to False')


def experiment(init):
    print('------------------------------------------')
    print('init={}'.format(init))
    print('------------------------------------------')

    # x
    input_size_a = Opt.subordinate_size * Opt.num_subordinate_cats_in_a
    input_size_b = Opt.subordinate_size * Opt.num_subordinate_cats_in_b
    input_size = input_size_a + input_size_b
    x = np.eye(input_size, dtype=np.float32)
    torch_x = torch.from_numpy(x.astype(np.float32))
    # y
    a_sub = np.hstack((np.eye(Opt.num_subordinate_cats_in_a).repeat(Opt.subordinate_size, axis=0),
                       np.zeros((input_size_a, Opt.num_subordinate_cats_in_a))))
    b_sub = np.hstack((np.zeros((input_size_b, Opt.num_subordinate_cats_in_b)),
                       np.eye(Opt.num_subordinate_cats_in_b).repeat(Opt.subordinate_size, axis=0)))
    a_sup = np.array([[1, 0]]).repeat(input_size_a, axis=0)
    b_sup = np.array([[0, 1]]).repeat(input_size_b, axis=0)
    sub_cols = np.vstack((a_sub, b_sub))
    sup_cols = np.vstack((a_sup, b_sup))
    y1 = np.hstack((sub_cols, np.zeros((sup_cols.shape[0], sup_cols.shape[1]))))
    y2 = np.hstack((np.zeros((sub_cols.shape[0], sub_cols.shape[1])), sup_cols))
    y = y1 + y2  # y1 has subordinate category feedback, y2 has superordinate category feedback
    torch_y = torch.from_numpy(y.astype(np.float32))

    # gold labels for category structure in A and B
    gold_sim_mat_a = np.rint(cosine_similarity(y[:input_size_a]))
    gold_sim_mat_b = np.rint(cosine_similarity(y[-input_size_b:]))

    # for probabilistic supervision - sample feedback from either y1 or y2
    y1_ids = np.arange(y1.shape[0])
    y2_ids = y1_ids + y1.shape[0]

    # net
    output_size = y.shape[1]
    if init == 'random':
        w1 = np.random.standard_normal(size=(input_size, Opt.hidden_size)) * Opt.scale_weights
    elif init == 'superordinate':
        w1 = make_superordinate_w1(Opt.hidden_size, input_size_a, input_size_b) * Opt.scale_weights
    elif init == 'identical':
        w1 = make_identical_w1(Opt.hidden_size, input_size_a, input_size_b) * Opt.scale_weights
    elif init == 'linear':  # each item is assigned same weight vector with linear transformation
        w1 = make_identical_w1(Opt.hidden_size, input_size_a, input_size_b) * Opt.scale_weights
        w1_delta = np.tile(np.linspace(0, 1, num=w1.shape[0])[:, np.newaxis], (1, w1.shape[1]))
        w1 += np.random.permutation(w1_delta)  # permute otherwise linear transform will align with category structure
    else:
        raise AttributeError('Invalid arg to "init".')
    net = Net(input_size=input_size,
              hidden_size=Opt.hidden_size,
              output_size=output_size,
              w1=w1,
              w2=np.random.standard_normal(size=(Opt.hidden_size, output_size)) * Opt.scale_weights)

    # optimizer + criterion
    optimizer = optim.SGD(net.parameters(), lr=Opt.lr)
    criterion = torch.nn.MSELoss()

    # eval before start of training
    net.eval()
    torch_o = net(torch_x)
    net.train()

    # train loop
    eval_score_a = EVAL_SCORE_A  # save time - don't evaluate ba after it has reached 1.0
    eval_score_b = EVAL_SCORE_B
    eval_epoch_idx = 0
    score_trajectory_a = np.ones(num_eval_epochs)
    score_trajectory_b = np.ones(num_eval_epochs)
    for i in range(Opt.num_epochs + 1):
        # eval
        if i % Opt.eval_interval == 0:
            print('Evaluating at epoch {}'.format(i))
            if Opt.representation == 'output':
                prediction_mat = torch_o.detach().numpy()
            elif Opt.representation == 'hidden':
                prediction_mat = net.linear1(torch_x).detach().numpy()
            else:
                raise AttributeError('Invalid arg to "REPRESENTATION".')
            sim_mat_a = cosine_similarity(prediction_mat[:input_size_a])
            sim_mat_b = cosine_similarity(prediction_mat[-input_size_b:])
            score_a = calc_cluster_score(sim_mat_a, gold_sim_mat_a, CLUSTER_METRIC) if eval_score_a else 1.0
            score_b = calc_cluster_score(sim_mat_b, gold_sim_mat_b, CLUSTER_METRIC) if eval_score_b else 1.0
            score_trajectory_a[eval_epoch_idx] = score_a
            score_trajectory_b[eval_epoch_idx] = score_b
            eval_epoch_idx += 1
            #
            print(prediction_mat[:input_size_a].round(3))
            print(prediction_mat[-input_size_b:].round(3))
            print(gold_sim_mat_a.round(1))
            print(sim_mat_a.round(4))
            print('{}_a={}'.format(CLUSTER_METRIC, score_a)) if EVAL_SCORE_A else None
            print('{}_b={}'.format(CLUSTER_METRIC, score_b)) if EVAL_SCORE_B else None
            print()
            #
            if score_a == 1.0:
                eval_score_a = False
            if score_b == 1.0:
                eval_score_b = False

        # flip superordinate feedback (all items are affected symmetrically)
        if bool(np.random.binomial(n=1, p=Opt.y2_flip_prob)):
            y2 = np.flipud(y2)  # flip vertically
            y = y1 + y2
            torch_y = torch.from_numpy(y.astype(np.float32))

        # shuffle superordinate feedback (items are not affected symmetrically)
        if bool(np.random.binomial(n=1, p=Opt.y2_shuffle_prob)):  # TODO make this a function of epoch id
            y2 = np.random.permutation(y2)
            y = y1 + y2
            torch_y = torch.from_numpy(y.astype(np.float32))

        # give feedback from either y1 or y2 but never together
        if Opt.separate_feedback[0]:
            row_ids = [idx1 if np.random.binomial(n=1, p=Opt.separate_feedback[1]) else idx2
                       for idx1, idx2 in zip(y1_ids, y2_ids)]
            y = np.vstack((y1, y2))[row_ids]  # select rows from y1 or y2
            torch_y = torch.from_numpy(y.astype(np.float32))

        # train
        optimizer.zero_grad()  # zero the gradient buffers
        torch_o = net(torch_x)
        loss = criterion(torch_o, torch_y)
        loss.backward()
        optimizer.step()  # update

        # mse
        if i % Opt.eval_interval == 0:
            mse = loss.detach().numpy().item()
            print('mse={}'.format(mse))

    return score_trajectory_a, score_trajectory_b


# get experimental results - returns tensor with shape = [NUM_REPS, 2, num_eval_epochs]
init_conditions = ['identical', 'linear', 'random']
results_1 = np.array([experiment(init=init_conditions[0]) for _ in range(Opt.num_reps)])
results_2 = np.array([experiment(init=init_conditions[1]) for _ in range(Opt.num_reps)])
results_3 = np.array([experiment(init=init_conditions[2]) for _ in range(Opt.num_reps)])

# plot results
if EVAL_SCORE_A and EVAL_SCORE_B:
    cat_names = ['A', 'B']
elif not EVAL_SCORE_A:
    cat_names = ['B']
elif not EVAL_SCORE_B:
    cat_names = ['A']
else:
    raise SystemExit('EVAL_SCORE_A and EVAL_SCORE_B are set to False')
#
for n, cat_name in enumerate(cat_names):
    # aggregate data
    xs = [eval_epochs] * len(init_conditions)
    ys = [results_1.mean(axis=0, keepdims=True)[:, n, :].squeeze(),
          results_2.mean(axis=0, keepdims=True)[:, n, :].squeeze(),
          results_3.mean(axis=0, keepdims=True)[:, n, :].squeeze()]
    stds = [results_1.std(axis=0, keepdims=True)[:, n, :].squeeze(),
            results_2.std(axis=0, keepdims=True)[:, n, :].squeeze(),
            results_3.std(axis=0, keepdims=True)[:, n, :].squeeze()]
    sems = [std / np.sqrt(Opt.num_reps) for std in stds]
    margins_of_error = [sem * t.ppf(1 - 0.05 / 2, Opt.num_reps - 1) for sem in sems]  # margins are 1/2 the total CI
    options = '\n'.join(['{}={}'.format(k, v) for k, v in sorted(Opt.__dict__.items())
                         if not k.startswith('_')])
    # fig
    fig = plot_trajectories(xs=xs,
                            ys=ys,
                            margins_of_error=margins_of_error,
                            options=options,
                            labels=init_conditions,
                            label_prefix='init = ',
                            name='category {} '.format(cat_name) + CLUSTER_METRIC,
                            ylim=[0.5, 1.05] if CLUSTER_METRIC in ['ba', 'fs'] else [0.0, 1.05],
                            figsize=(6, 6))
    fig.show()
