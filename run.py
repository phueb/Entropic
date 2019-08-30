import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.net import NeuralNet
from src.eval import calc_cluster_score
from src.plot import plot_trajectories

VERBOSE = False

NUM_REPS = 10
NUM_HIDDEN = 8
NUM_EPOCHS = 50
EVAL_INTERVAL = 10
CLUSTER_METRIC = 'ba'

ALL_SAME = False
# TODO is ba better because weights for A members are the same or because of distance between cat A and B?

eval_epochs = np.arange(0, NUM_EPOCHS, EVAL_INTERVAL)
num_eval_epochs = len(eval_epochs)


def experiment(smart_init):
    print('------------------------------------------')
    print('smart_init={}'.format(smart_init))
    print('------------------------------------------')

    x = np.array(([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1
                  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 2
                  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 3
                  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4
                  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 5
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 6
                  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 7
                  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 8
                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 9
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 10
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 11
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 12
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 13
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 14
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # 15
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # 16
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 17
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),  # 18
                 dtype=np.float32)
    y = np.array(([1, 0, 0, 0, 0, 0, 1, 0],
                  [1, 0, 0, 0, 0, 0, 1, 0],
                  [1, 0, 0, 0, 0, 0, 1, 0],
                  [0, 1, 0, 0, 0, 0, 1, 0],
                  [0, 1, 0, 0, 0, 0, 1, 0],
                  [0, 1, 0, 0, 0, 0, 1, 0],
                  [0, 0, 1, 0, 0, 0, 1, 0],
                  [0, 0, 1, 0, 0, 0, 1, 0],
                  [0, 0, 1, 0, 0, 0, 1, 0],
                  [0, 0, 0, 1, 0, 0, 0, 1],
                  [0, 0, 0, 1, 0, 0, 0, 1],
                  [0, 0, 0, 1, 0, 0, 0, 1],
                  [0, 0, 0, 0, 1, 0, 0, 1],
                  [0, 0, 0, 0, 1, 0, 0, 1],
                  [0, 0, 0, 0, 1, 0, 0, 1],
                  [0, 0, 0, 0, 0, 1, 0, 1],
                  [0, 0, 0, 0, 0, 1, 0, 1],
                  [0, 0, 0, 0, 0, 1, 0, 1]),
                 dtype=np.float32)

    # to torch
    torch_x = torch.from_numpy(x)
    torch_y = torch.from_numpy(y)

    gold_mat_a = np.matmul(y[:9], y[:9].T) - 1
    gold_mat_b = np.matmul(y[9:], y[9:].T) - 1

    # net
    a = np.tile(np.random.standard_normal(size=(1, NUM_HIDDEN)), (9, 1))  # same weights for inputs in cat A
    if ALL_SAME:
        b = a
    else:
        b = np.tile(np.random.standard_normal(size=(1, NUM_HIDDEN)), (9, 1))  # same weights for inputs in cat B
    w1 = np.vstack((a, b))  # [18, NUM_HIDDEN]
    net = NeuralNet(num_in=18,
                    num_hidden=NUM_HIDDEN,
                    num_out=8,
                    w1=w1 if smart_init else None)
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
            prediction_mat = net.predict(x)
            sim_mat_a = cosine_similarity(prediction_mat[:9])
            sim_mat_b = cosine_similarity(prediction_mat[9:])
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
        o = net.forward(torch_x)
        net.backward(torch_x, torch_y, o)

    # mse
    mse = torch.mean((torch_y - net(torch_x)) ** 2).detach().item()
    print('mse={}'.format(mse))

    return score_trajectory_a, score_trajectory_b


# get experimental results
results_1 = np.array([experiment(smart_init=True) for _ in range(NUM_REPS)])  # [NUM_REPS, 2, num_eval_epochs]
results_2 = np.array([experiment(smart_init=False) for _ in range(NUM_REPS)])  # [NUM_REPS, 2, num_eval_epochs]

# plot results
cat_names = ['A', 'B']
xs = [eval_epochs] * 2
for n, cat_name in enumerate(cat_names):
    ys = [results_1.mean(axis=0, keepdims=True)[:, n, :].squeeze(),
          results_2.mean(axis=0, keepdims=True)[:, n, :].squeeze()]
    fig = plot_trajectories(xs=xs,
                            ys=ys,
                            labels=[True, False],
                            label_prefix='smart init = ',
                            name='category {} '.format(cat_name) + CLUSTER_METRIC)
    fig.show()
