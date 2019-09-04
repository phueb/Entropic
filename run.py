import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.net import NeuralNet
from src.eval import calc_cluster_score
from src.plot import plot_trajectories
from src.utils import make_superordinate_w1, make_identical_w1

VERBOSE = False

NUM_REPS = 10
NUM_HIDDEN = 8
NUM_EPOCHS = 50
EVAL_INTERVAL = 10
CLUSTER_METRIC = 'ba'

NUM_SUBORDINATE_CATS_IN_A = 3
NUM_SUBORDINATE_CATS_IN_B = 3
SUBORDINATE_SIZE = 3

"""
structure of the data:
there are 2 superordinate categories (category A and B).
each superordinate category contains NUM_SUBORDINATES subordinate categories.
each subordinate category contains SUBORDINATE_SIZE items.
"""

eval_epochs = np.arange(0, NUM_EPOCHS, EVAL_INTERVAL)
num_eval_epochs = len(eval_epochs)


def experiment(init):
    print('------------------------------------------')
    print('init={}'.format(init))
    print('------------------------------------------')

    # x
    num_items_in_a = SUBORDINATE_SIZE * NUM_SUBORDINATE_CATS_IN_A
    num_items_in_b = SUBORDINATE_SIZE * NUM_SUBORDINATE_CATS_IN_B
    num_x = num_items_in_a + num_items_in_b
    x = np.eye(num_x, dtype=np.float32)
    # y
    a_sub = np.hstack((np.eye(NUM_SUBORDINATE_CATS_IN_A).repeat(SUBORDINATE_SIZE, axis=0),
                       np.zeros((num_items_in_a, NUM_SUBORDINATE_CATS_IN_A))))
    b_sub = np.hstack((np.zeros((num_items_in_b, NUM_SUBORDINATE_CATS_IN_B)),
                       np.eye(NUM_SUBORDINATE_CATS_IN_B).repeat(SUBORDINATE_SIZE, axis=0)))
    a_sup = np.array([[1, 0]]).repeat(num_items_in_a, axis=0)
    b_sup = np.array([[0, 1]]).repeat(num_items_in_a, axis=0)
    sub_cols = np.vstack((a_sub, b_sub))
    sup_cols = np.vstack((a_sup, b_sup))
    y = np.hstack((sub_cols, sup_cols))
    y = y.astype(np.float32)
    print(y)

    # to torch
    torch_x = torch.from_numpy(x)
    torch_y = torch.from_numpy(y)

    gold_mat_a = np.matmul(y[:num_items_in_a], y[:num_items_in_a].T) - 1
    gold_mat_b = np.matmul(y[num_items_in_a:], y[num_items_in_a:].T) - 1

    # net
    if init == 'random':
        w1 = None
    elif init == 'superordinate':
        w1 = make_superordinate_w1(NUM_HIDDEN, num_items_in_a, num_items_in_b)
    elif init == 'identical':
        w1 = make_identical_w1(NUM_HIDDEN, num_items_in_a, num_items_in_b)
    else:
        raise AttributeError('Invalid arg to "init".')
    num_items_total = num_items_in_a + num_items_in_b
    net = NeuralNet(num_in=num_items_total,
                    num_hidden=NUM_HIDDEN,
                    num_out=y.shape[1],
                    w1=w1)

    # train loop
    eval_score_a = True  # save time - don't evaluate ba after it has reached 1.0
    eval_score_b = True
    eval_epoch_idx = 0
    score_trajectory_a = np.ones(num_eval_epochs)
    score_trajectory_b = np.ones(num_eval_epochs)
    for i in range(NUM_EPOCHS):
        # eval
        if i % EVAL_INTERVAL == 0:
            print('Evaluating at epoch {}'.format(i))
            prediction_mat = net(torch_x).numpy()
            sim_mat_a = cosine_similarity(prediction_mat[:num_items_in_a])
            sim_mat_b = cosine_similarity(prediction_mat[num_items_in_a:])
            score_a = calc_cluster_score(sim_mat_a, gold_mat_a, CLUSTER_METRIC) if eval_score_a else 1.0
            score_b = calc_cluster_score(sim_mat_b, gold_mat_b, CLUSTER_METRIC) if eval_score_b else 1.0
            score_trajectory_a[eval_epoch_idx] = score_a
            score_trajectory_b[eval_epoch_idx] = score_b
            eval_epoch_idx += 1
            #
            print(sim_mat_a.round(1))
            print('{}_a={}'.format(CLUSTER_METRIC, score_a))
            print('{}_b={}'.format(CLUSTER_METRIC, score_b))
            print()
            #
            if score_a == 1.0:
                eval_score_a = False
            if score_b == 1.0:
                eval_score_b = False
        # train
        o = net(torch_x)
        net.backward(torch_x, torch_y, o)

    # mse
    mse = torch.mean((torch_y - net(torch_x)) ** 2).detach().item()
    print('mse={}'.format(mse))

    return score_trajectory_a, score_trajectory_b


# get experimental results returns tensor with shape = [NUM_REPS, 2, num_eval_epochs]
init_conditions = ['identical', 'superordinate', 'random']
results_1 = np.array([experiment(init=init_conditions[0]) for _ in range(NUM_REPS)])
results_2 = np.array([experiment(init=init_conditions[1]) for _ in range(NUM_REPS)])
results_3 = np.array([experiment(init=init_conditions[2]) for _ in range(NUM_REPS)])

# plot results
cat_names = ['A', 'B']
for n, cat_name in enumerate(cat_names):
    # aggregate data
    xs = [eval_epochs] * len(init_conditions)
    ys = [results_1.mean(axis=0, keepdims=True)[:, n, :].squeeze(),
          results_2.mean(axis=0, keepdims=True)[:, n, :].squeeze(),
          results_3.mean(axis=0, keepdims=True)[:, n, :].squeeze()]

    # TODO add confidence-interval

    fig = plot_trajectories(xs=xs,
                            ys=ys,
                            title='n={}'.format(NUM_REPS),
                            labels=init_conditions,
                            label_prefix='init = ',
                            name='category {} '.format(cat_name) + CLUSTER_METRIC)
    fig.show()
