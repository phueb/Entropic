import numpy as np
from pyitlib import discrete_random_variable as drv

VOCAB_SIZE = 4096

p = np.random.random(VOCAB_SIZE)
q = np.random.random(VOCAB_SIZE)
p = p / p.sum()
q = q / q.sum()
# print(drv.entropy_cross_pmf(p, q, base=np.exp(1)))
# print(drv.entropy_cross_pmf(p, p, base=np.exp(1)))


# is there more error when W word cues the superordinate category or the target category?
e = 1e-12
p_ = np.clip([1.0, 0.0, 0.0, 0.0, 0.00, 0.00, 0.00, 0.00], 0, 1-e) + e
q1 = np.clip([0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.25], 0, 1-e) + e
q2 = np.clip([0.0, 0.0, 0.0, 0.0, 0.00, 0.00, 0.50, 0.50], 0, 1-e) + e
print(drv.entropy_cross_pmf(p_, q1))
print(drv.entropy_cross_pmf(p_, q2))
