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
                 doc_size: int = 400_000,
                 delay: int = 200_000,
                 num_types: int = 1024,
                 num_fragments: int = 4,  # number of categories in xws
                 period_probability: Tuple[float, float] = (0.0, 0.0),
                 alpha: float = 2.0,
                 num_sentinels: int = 0,
                 seed: Optional[int] = None,
                 ) -> None:
        self.doc_size = doc_size
        self.num_types = num_types
        self.num_fragments = num_fragments
        self.period_probability = period_probability
        self.alpha = alpha
        self.delay = delay
        self.num_sentinels = num_sentinels

        self.num_v = self.num_types // 4
        self.num_w = self.num_types // 4
        self.num_x = self.num_types // 4
        self.num_y = self.num_types // 4

        self.v = [f'v{i:0>6}' for i in range(self.num_v)]
        self.w = [f'w{i:0>6}' for i in range(self.num_w)]
        self.x = [f'x{i:0>6}' for i in range(self.num_x)]
        self.y = [f'y{i:0>6}' for i in range(self.num_y)]

        self.types = self.v + self.w + self.x + self.y

        # assign x-words to categories
        self.xi2cat_id = {xi: cat_id for xi, cat_id in zip(self.x, cycle(range(self.num_fragments)))}
        self.cat_id2x = {frag_id: [xi for xi, cat_id in self.xi2cat_id.items() if cat_id == frag_id]
                         for frag_id in range(self.num_fragments)}

        # map subsets of xis to mutually exclusive subsets/fragments of y
        w_fragments = [self.w[offset::num_fragments] for offset in range(num_fragments)]
        y_fragments = [self.y[offset::num_fragments] for offset in range(num_fragments)]
        self.xi2w = {xi: w_fragments[self.xi2cat_id[xi]] for xi in self.x}
        self.xi2y = {xi: y_fragments[self.xi2cat_id[xi]] for xi in self.x}

        # check
        w_fragment_size = self.num_w // num_fragments
        x_fragment_size = self.num_x // num_fragments
        y_fragment_size = self.num_y // num_fragments
        for xi, w in self.xi2w.items():
            assert len(w) == w_fragment_size, (len(w), w_fragment_size)
        for xi, y in self.xi2y.items():
            assert len(y) == y_fragment_size, (len(y), y_fragment_size)

        assert self.num_sentinels <= x_fragment_size, f'"num_sentinels" must be <= {x_fragment_size}'

        non_sentinels = self.cat_id2x[self.num_fragments - 1][num_sentinels:]
        self.x_without_non_sentinels = [xi for xi in self.x if xi not in non_sentinels]

        # the number of legal joint outcomes is the total number divided by the fragment size
        self.num_possible = self.num_x * self.num_y / num_fragments

        print('Initialized ToyCorpus')

        if seed is not None:
            random.seed(seed)

    @cached_property
    def doc(self) -> str:
        joint_outcomes = set()

        # make pseudo_periods
        pseudo_periods = []
        c = cycle(range(self.num_fragments))
        yw_fragments = [self.y[offset::self.num_fragments] for offset in range(self.num_fragments)]
        num_max = 8  # should be small - to ensure that joint entropy is smaller in partition 1
        for yw_pop in list(zip(*yw_fragments))[:num_max]:
            i = next(c)
            pseudo_periods.append(yw_pop[i])

        # make cumulative weights over y-words that mimic power distribution
        logits = [(xi + 1) ** self.alpha for xi in range(len(pseudo_periods))]
        cum_weights = [l / logits[-1] for l in logits]

        res = ''
        num_words_in_window = 4
        for n in range(self.doc_size // num_words_in_window):

            # corpus behaves differently before and after delay
            if n * num_words_in_window > self.delay:
                x = self.x
                period_probability = self.period_probability[1]
            else:
                x = self.x_without_non_sentinels
                period_probability = self.period_probability[0]

            # sample xi randomly
            xi = random.choice(x)

            # sample vi randomly
            vi = random.choice(self.v)

            # sample wi consistent with xi
            wi = random.choice(self.xi2w[xi])  # TODO add option to insert pseudo-period here too

            # sample yi that is consistent with ALL xi categories (e.g. PERIOD)
            if random.random() < period_probability:
                yi = random.choices(pseudo_periods, cum_weights=cum_weights, k=1)[0]
            # sample yi consistent with xi category
            else:
                yi = random.choice(self.xi2y[xi])

            # collect
            res += f'{vi} {wi} {xi} {yi} '  # whitespace after each
            joint_outcomes.add((vi, wi, xi, yi))

        print(f'Number of unique joint outcomes={len(joint_outcomes):,}/{self.num_possible:,}')
        print(f'Coverage={len(joint_outcomes) / self.num_possible:.2f}')

        return res

    @cached_property
    def sim_mat_gold(self) -> np.ndarray:

        # every xi is related to every other xi, depending on num_fragments
        res = np.zeros((self.num_x, self.num_x))
        for row_id in range(self.num_x):
            offset = row_id % self.num_fragments
            res[row_id, offset::self.num_fragments] += 1

        return res