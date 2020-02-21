import numpy as np
from pyitlib import discrete_random_variable as drv

VOCAB_SIZE = 4096

p = np.random.random(VOCAB_SIZE)
q = np.random.random(VOCAB_SIZE)
p = p / p.sum()
q = q / q.sum()
xe = drv.entropy_cross_pmf(p, q, base=np.exp(1))
print(xe)
xe = drv.entropy_cross_pmf(p, p, base=np.exp(1))
print(xe)