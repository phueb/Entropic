from cached_property import cached_property
import random
from itertools import cycle
import numpy as np
from typing import Tuple, Optional


class Corpus:
    """
    methods for making a document, a string of artificial words following the structure (V, W, X, Y).
    """

    def __init__(self,
                 doc_size: int,
                 delay: int,
                 num_types: int,
                 num_fragments: int,
                 period_probability: Tuple[float, float],
                 num_sentinels: int,
                 sample_w: str,
                 sample_v: str,
                 alpha: float = 2.0,
                 seed: Optional[int] = None,
                 ) -> None:
        self.doc_size = doc_size
        self.num_types = num_types
        self.num_fragments = num_fragments
        self.period_probability = period_probability
        self.sample_w = sample_w
        self.sample_v = sample_v
        self.alpha = alpha
        self.delay = delay
        self.num_sentinels = num_sentinels

        self.num_words_in_window = 4
        self.slots = ['v', 'w', 'x', 'y']

        self.num_v = self.num_types // self.num_words_in_window
        self.num_w = self.num_types // self.num_words_in_window
        self.num_x = self.num_types // self.num_words_in_window
        self.num_y = self.num_types // self.num_words_in_window

        self.v = [f'{self.slots[0]}{i:0>6}' for i in range(self.num_v)]
        self.w = [f'{self.slots[1]}{i:0>6}' for i in range(self.num_w)]
        self.x = [f'{self.slots[2]}{i:0>6}' for i in range(self.num_x)]
        self.y = [f'{self.slots[3]}{i:0>6}' for i in range(self.num_y)]

        self.types = self.v + self.w + self.x + self.y
        assert len(self.types) == num_types

        # assign x-words to categories
        self.xi2cat_id = {xi: cat_id for xi, cat_id in zip(self.x, cycle(range(self.num_fragments)))}
        self.cat_id2x = {frag_id: [xi for xi, cat_id in self.xi2cat_id.items() if cat_id == frag_id]
                         for frag_id in range(self.num_fragments)}

        # map subsets of xis to mutually exclusive subsets/fragments
        v_fragments = [self.v[offset::num_fragments] for offset in range(num_fragments)]
        w_fragments = [self.w[offset::num_fragments] for offset in range(num_fragments)]
        y_fragments = [self.y[offset::num_fragments] for offset in range(num_fragments)]
        self.xi2v = {xi: v_fragments[self.xi2cat_id[xi]] for xi in self.x}
        self.xi2w = {xi: w_fragments[self.xi2cat_id[xi]] for xi in self.x}
        self.xi2y = {xi: y_fragments[self.xi2cat_id[xi]] for xi in self.x}

        self.xi2vi = {xi: vi for xi, vi in zip(self.x, self.v)}
        self.xi2wi = {xi: wi for xi, wi in zip(self.x, self.w)}

        # check
        x_fragment_size = self.num_x // num_fragments
        assert 0 < self.num_sentinels <= x_fragment_size, f'Check that 0 < "num_sentinels"  <= {x_fragment_size}'

        non_sentinels = []
        for cat_id in range(self.num_fragments):
            non_sentinels += self.cat_id2x[cat_id][num_sentinels:]
        self.x_without_non_sentinels = [xi for xi in self.x if xi not in non_sentinels]

        # the number of legal joint outcomes is the total number divided by the fragment size
        self.num_possible_x_y = self.num_x * self.num_y / num_fragments

        print('Initialized ToyCorpus')

        if seed is not None:
            random.seed(seed)

    @cached_property
    def doc(self) -> str:
        joint_x_y_outcomes = set()

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
        for n in range(self.doc_size // self.num_words_in_window):

            # corpus behaves differently before and after delay
            if n * self.num_words_in_window > self.delay:
                x_population = self.x
                period_probability = self.period_probability[1]
            else:
                x_population = self.x_without_non_sentinels
                period_probability = self.period_probability[0]

            # sample xi systematically
            xi = random.choice(x_population)  # do not sample from itertools.cycle because of predictable structure

            # sample vi
            if self.sample_v == 'item':
                vi = self.xi2vi[xi]
            elif self.sample_v == 'target-category':
                vi = random.choice(self.xi2v[xi])
            elif self.sample_v == 'superordinate':
                vi = random.choice(self.v)
            else:
                raise AttributeError('Invalid arg to "sample_v".')

            # sample wi
            if self.sample_w == 'item':
                wi = self.xi2wi[xi]
            elif self.sample_w == 'target-category':
                wi = random.choice(self.xi2w[xi])
            elif self.sample_w == 'superordinate':
                wi = random.choice(self.w)
            else:
                raise AttributeError('Invalid arg to "sample_w".')

            # sample yi that is consistent with all x categories (e.g. PERIOD)
            if random.random() < period_probability:
                yi = random.choices(pseudo_periods, cum_weights=cum_weights, k=1)[0]
            # sample yi consistent with xi category
            else:
                yi = random.choice(self.xi2y[xi])

            # collect
            res += f'{vi} {wi} {xi} {yi} '  # whitespace after each
            joint_x_y_outcomes.add((xi, yi))

        print(f'Number of unique joint (x,y) outcomes={len(joint_x_y_outcomes):,}/{self.num_possible_x_y:,}')
        print(f'Coverage={len(joint_x_y_outcomes) / self.num_possible_x_y:.2f}')

        return res

    @cached_property
    def sim_mat_gold(self) -> np.ndarray:

        # every xi is related to every other xi, depending on num_fragments
        res = np.zeros((self.num_x, self.num_x))
        for row_id in range(self.num_x):
            offset = row_id % self.num_fragments
            res[row_id, offset::self.num_fragments] += 1

        return res