from cached_property import cached_property
import random
from itertools import cycle
import numpy as np
from typing import Tuple, Optional


class ToyCorpus:
    """
    methods for making a document, a string of artificial words following the structure (xw, yw, xw, yw, ..).
    example document: "x1 y5 x34 y82 x93 y3 x45 y11".
    """

    def __init__(self,
                 doc_size: int = 200_000,
                 num_types: int = 1024,
                 num_xws: int = 512,
                 num_fragments: int = 4,  # number of categories in xws
                 period_probability: Tuple[float, float] = (0.1, 0.0),
                 alpha: float = 2.0,
                 delay: int = 50_000,
                 seed: Optional[int] = None,
                 ) -> None:
        self.doc_size = doc_size
        self.num_types = num_types
        self.num_xws = num_xws
        self.num_fragments = num_fragments
        self.period_probability = period_probability
        self.alpha = alpha
        self.delay = delay
        self.num_yws = self.num_types - self.num_xws

        self.xws = [f'x{i:0>6}' for i in range(self.num_xws)]
        self.yws = [f'y{i:0>6}' for i in range(self.num_yws)]

        # map subsets of xws to mutually exclusive subsets/fragments of yws
        yw_fragments = [self.yws[offset::num_fragments] for offset in range(num_fragments)]
        c = cycle(yw_fragments)
        self.xw2yws = {xw: next(c) for xw in self.xws}

        # the number of legal joint outcomes is the total number divided by the fragment size
        self.num_possible = self.num_xws*self.num_yws / num_fragments

        # set aside x-words belonging to last category, to be introduced after a delay
        self.xws_without_last_category = [xw for xw in self.xws
                                          if self.xw2yws[xw] != yw_fragments[-1]]
        print(len(self.xws))
        print(len(self.xws_without_last_category))

        # x-words whose representations are used for evaluation
        self.eval_xws = self.xws[:self.num_fragments]

        print('Initialized ToyCorpus')
        print(f'Lowest theoretical pp ={self.num_yws // self.num_fragments:>6,}')
        print(f'Number of y-word types={self.num_yws:>6,}')

        if seed is not None:
            random.seed(seed)

    @cached_property
    def doc(self) -> str:
        joint_outcomes = set()

        # make pseudo_periods
        pseudo_periods = []
        c = cycle(range(self.num_fragments))
        yw_fragments = [self.yws[offset::self.num_fragments] for offset in range(self.num_fragments)]
        num_max = 8  # should be small - to ensure that joint entropy is smaller in partition 1
        for yw_pop in list(zip(*yw_fragments))[:num_max]:
            i = next(c)
            pseudo_periods.append(yw_pop[i])

        # make cumulative weights that mimic power distribution
        logits = [(xi + 1) ** self.alpha for xi in range(len(pseudo_periods))]
        cum_weights = [l / logits[-1] for l in logits]

        res = ''
        for n in range(self.doc_size // 2):  # divide by 2 because each loop adds 2 words

            # corpus behaves differently before and after delay
            if n * 2 > self.delay:
                xws = self.xws
                period_probability = self.period_probability[1]
            else:
                xws = self.xws_without_last_category
                period_probability = self.period_probability[0]

            # sample xw randomly
            xw = random.choice(xws)

            # sample yw that is consistent with ALL xw categories (e.g. PERIOD)
            if random.random() < period_probability and xw not in self.eval_xws:
                yw = random.choices(pseudo_periods, cum_weights=cum_weights, k=1)[0]

            # sample yw consistent with xw category
            else:
                yw = random.choice(self.xw2yws[xw])

            # collect
            res += f'{xw} {yw} '  # whitespace after each
            joint_outcomes.add((xw, yw))

        print(f'Number of unique joint outcomes={len(joint_outcomes):,}/{self.num_possible:,}')
        print(f'Coverage={len(joint_outcomes) / self.num_possible:.2f}')

        return res

    @cached_property
    def sim_mat_gold(self) -> np.ndarray:

        # every xw is related to every other xw, depending on num_fragments
        res = np.zeros((self.num_xws, self.num_xws))
        for row_id in range(self.num_xws):
            offset = row_id % self.num_fragments
            res[row_id, offset::self.num_fragments] += 1

        return res