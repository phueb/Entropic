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

SCALE_WEIGHTS = 1.0  # works with 1.0 but not with 0.01 or 0.1
LEARNING_RATE = 1.0  # TODO test
HIDDEN_SIZE = 4
NUM_REPS = 3
NUM_EPOCHS = 1000
EVAL_INTERVAL = 100
CLUSTER_METRIC = 'ba'

NUM_SUBORDINATE_CATS_IN_A = 3
NUM_SUBORDINATE_CATS_IN_B = 3
SUBORDINATE_SIZE = 3

EVAL_SCORE_A = True
EVAL_SCORE_B = False  # False to save time

"""
structure of the data:
there are 2 superordinate categories (category A and B).
each superordinate category contains NUM_SUBORDINATES subordinate categories.
each subordinate category contains SUBORDINATE_SIZE items.
"""

eval_epochs = np.arange(0, NUM_EPOCHS, EVAL_INTERVAL)
num_eval_epochs = len(eval_epochs)

if not EVAL_SCORE_A and not EVAL_SCORE_B:
    raise SystemExit('EVAL_SCORE_A and EVAL_SCORE_B are set to False')


def experiment(init):
    print('------------------------------------------')
    print('init={}'.format(init))
    print('------------------------------------------')

    # x
    input_size_a = SUBORDINATE_SIZE * NUM_SUBORDINATE_CATS_IN_A
    input_size_b = SUBORDINATE_SIZE * NUM_SUBORDINATE_CATS_IN_B
    input_size = input_size_a + input_size_b
    x = np.eye(input_size, dtype=np.float32)
    # y
    a_sub = np.hstack((np.eye(NUM_SUBORDINATE_CATS_IN_A).repeat(SUBORDINATE_SIZE, axis=0),
                       np.zeros((input_size_a, NUM_SUBORDINATE_CATS_IN_A))))
    b_sub = np.hstack((np.zeros((input_size_b, NUM_SUBORDINATE_CATS_IN_B)),
                       np.eye(NUM_SUBORDINATE_CATS_IN_B).repeat(SUBORDINATE_SIZE, axis=0)))
    a_sup = np.array([[1, 0]]).repeat(input_size_a, axis=0)
    b_sup = np.array([[0, 1]]).repeat(input_size_b, axis=0)
    sub_cols = np.vstack((a_sub, b_sub))
    sup_cols = np.vstack((a_sup, b_sup))
    y = np.hstack((sub_cols, sup_cols))
    #
    torch_x = torch.from_numpy(x.astype(np.float32))
    torch_y = torch.from_numpy(y.astype(np.float32))

    # gold labels for category structure in A and B
    gold_sim_mat_a = np.rint(cosine_similarity(y[:input_size_a]))
    gold_sim_mat_b = np.rint(cosine_similarity(y[-input_size_b:]))

    # net

    # TODO use 2 layer network

    output_size = y.shape[1]
    if init == 'random':
        w1 = np.random.standard_normal(size=(input_size, HIDDEN_SIZE)) * SCALE_WEIGHTS
    elif init == 'superordinate':
        w1 = make_superordinate_w1(HIDDEN_SIZE, input_size_a, input_size_b) * SCALE_WEIGHTS
    elif init == 'identical':
        w1 = make_identical_w1(HIDDEN_SIZE, input_size_a, input_size_b) * SCALE_WEIGHTS
    else:
        raise AttributeError('Invalid arg to "init".')
    net = Net(input_size=input_size,
              hidden_size=HIDDEN_SIZE,
              output_size=output_size,
              w1=w1,
              w2=np.random.standard_normal(size=(HIDDEN_SIZE, output_size)) * SCALE_WEIGHTS)

    # optimizer + criterion
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE)
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
    for i in range(NUM_EPOCHS):
        # eval
        if i % EVAL_INTERVAL == 0:
            print('Evaluating at epoch {}'.format(i))
            prediction_mat = torch_o.detach().numpy()
            sim_mat_a = cosine_similarity(prediction_mat[:input_size_a])
            sim_mat_b = cosine_similarity(prediction_mat[-input_size_b:])
            score_a = calc_cluster_score(sim_mat_a, gold_sim_mat_a, CLUSTER_METRIC) if eval_score_a else 1.0
            score_b = calc_cluster_score(sim_mat_b, gold_sim_mat_b, CLUSTER_METRIC) if eval_score_b else 1.0
            score_trajectory_a[eval_epoch_idx] = score_a
            score_trajectory_b[eval_epoch_idx] = score_b
            eval_epoch_idx += 1
            #
            print(prediction_mat.round(2))
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

        # train
        optimizer.zero_grad()  # zero the gradient buffers
        torch_o = net(torch_x)
        loss = criterion(torch_o, torch_y)
        loss.backward()
        optimizer.step()  # update

        # mse
        if i % EVAL_INTERVAL == 0:
            mse = loss.detach().numpy().item()
            print('mse={}'.format(mse))

    return score_trajectory_a, score_trajectory_b


# get experimental results - returns tensor with shape = [NUM_REPS, 2, num_eval_epochs]
init_conditions = ['identical', 'superordinate', 'random']
results_1 = np.array([experiment(init=init_conditions[0]) for _ in range(NUM_REPS)])
results_2 = np.array([experiment(init=init_conditions[1]) for _ in range(NUM_REPS)])
results_3 = np.array([experiment(init=init_conditions[2]) for _ in range(NUM_REPS)])

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
    sems = [std / np.sqrt(NUM_REPS) for std in stds]
    cis = [sem * t.ppf((1 + 0.95) / 2, n-1) for sem in sems]
    fig = plot_trajectories(xs=xs,
                            ys=ys,
                            cis=stds,
                            title='n={}'.format(NUM_REPS),
                            labels=init_conditions,
                            label_prefix='init = ',
                            name='category {} '.format(cat_name) + CLUSTER_METRIC,
                            ylim=[0.5, 1.0] if CLUSTER_METRIC in ['ba', 'fs'] else [0.0, 1.0])
    fig.show()
