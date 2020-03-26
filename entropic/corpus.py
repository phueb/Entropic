from cached_property import cached_property
import random
from itertools import cycle
import numpy as np
from typing import Tuple, Optional, List


class Corpus:
    """
    methods for making a document, a string of artificial words following the structure (A, X, B, Y).
    """

    def __init__(self,
                 doc_size: int,
                 delay: int,
                 num_types: int,
                 num_fragments: int,
                 starvation: Tuple[float, float],
                 num_sentinels: int,
                 sample_b: Tuple[str, str],
                 sample_a: Tuple[str, str],
                 alpha: float = 2.0,
                 seed: Optional[int] = None,
                 ) -> None:
        self.doc_size = doc_size
        self.num_types = num_types
        self.num_fragments = num_fragments
        self.starvation = starvation
        self.sample_b = sample_b
        self.sample_a = sample_a
        self.alpha = alpha
        self.delay = delay
        self.num_sentinels = num_sentinels

        self.num_words_in_window = 4
        self.slots = ['a', 'x', 'b', 'y']

        self.num_a = self.num_types // self.num_words_in_window
        self.num_x = self.num_types // self.num_words_in_window
        self.num_b = self.num_types // self.num_words_in_window
        self.num_y = self.num_types // self.num_words_in_window

        self.a = [f'{self.slots[0]}{i:0>6}' for i in range(self.num_a)]
        self.x = [f'{self.slots[1]}{i:0>6}' for i in range(self.num_x)]
        self.b = [f'{self.slots[2]}{i:0>6}' for i in range(self.num_b)]
        self.y = [f'{self.slots[3]}{i:0>6}' for i in range(self.num_y)]

        self.types = self.a + self.b + self.x + self.y  # order alphabetically
        assert len(self.types) == num_types

        # assign x-words to categories
        self.xi2cat_id = {xi: cat_id for xi, cat_id in zip(self.x, cycle(range(self.num_fragments)))}
        self.cat_id2x = {frag_id: [xi for xi, cat_id in self.xi2cat_id.items() if cat_id == frag_id]
                         for frag_id in range(self.num_fragments)}

        # map subsets of xis to mutually exclusive subsets/fragments
        a_fragments = [self.a[offset::num_fragments] for offset in range(num_fragments)]
        b_fragments = [self.b[offset::num_fragments] for offset in range(num_fragments)]
        y_fragments = [self.y[offset::num_fragments] for offset in range(num_fragments)]
        self.xi2a = {xi: a_fragments[self.xi2cat_id[xi]] for xi in self.x}
        self.xi2b = {xi: b_fragments[self.xi2cat_id[xi]] for xi in self.x}
        self.xi2y = {xi: y_fragments[self.xi2cat_id[xi]] for xi in self.x}

        self.xi2ai = {xi: ai for xi, ai in zip(self.x, self.a)}
        self.xi2bi = {xi: bi for xi, bi in zip(self.x, self.b)}

        # check
        x_fragment_size = self.num_x // num_fragments
        assert 0 < self.num_sentinels <= x_fragment_size, f'Check that 0 < "num_sentinels"  <= {x_fragment_size}'

        non_sentinels = []
        for cat_id in range(self.num_fragments):
            non_sentinels += self.cat_id2x[cat_id][num_sentinels:]
        self.x1 = [xi for xi in self.x if xi not in non_sentinels]  # during delay
        self.x2 = self.x                                            # after delay
        print('num x1', len(self.x1))
        print('num x2', len(self.x2))


        if seed is not None:
            random.seed(seed)

    @cached_property
    def doc(self) -> str:

        doc_size1 = self.delay
        doc_size2 = self.doc_size - self.delay

        doc1 = self.make_doc(self.x1, doc_size1, self.starvation[0], self.sample_a[0], self.sample_b[0])
        doc2 = self.make_doc(self.x2, doc_size2, self.starvation[1], self.sample_a[1], self.sample_b[1])
        return doc1 + doc2

    def make_doc(self,
                 x: List[str],
                 doc_size: int,
                 starvation: float,
                 sample_a: str,
                 sample_b: str,
                 ) -> str:

        res = ''
        for n in range(doc_size // self.num_words_in_window):

            # sample xi
            xi = random.choice(x)  # do not sample from itertools.cycle because of predictable structure

            # sample ai
            # TODO make a condition in which Ai is non-compositional, "BRIGHT person thinks"
            #  essentially, pick an Ai that is incongruent or not in the correct semantic category
            #  that would be part 3 of paper 3
            if sample_a == 'item':
                ai = self.xi2ai[xi]
            elif sample_a == 'sub':
                ai = random.choice(self.xi2a[xi])
            elif sample_a == 'super':
                ai = random.choice(self.a)
            else:
                raise AttributeError('Invalid arg to "sample_a".')

            # sample bi
            if sample_b == 'item':
                bi = self.xi2bi[xi]
            elif sample_b == 'sub':
                bi = random.choice(self.xi2b[xi])
            elif sample_b == 'super':
                bi = random.choice(self.b)
            else:
                raise AttributeError('Invalid arg to "sample_b".')

            # sample yi
            if random.random() < starvation:
                yi = random.choice(self.random_periods)
            else:
                yi = random.choice(self.xi2y[xi])

            # collect
            res += f'{ai} {xi} {bi} {yi} '  # whitespace after each

        return res

    @cached_property
    def random_periods(self) -> List[str]:
        periods = []
        c = cycle(range(self.num_fragments))
        yw_fragments = [self.y[offset::self.num_fragments] for offset in range(self.num_fragments)]
        num_max = 8  # should be small - to ensure that joint entropy is smaller in partition 1
        for yw_pop in list(zip(*yw_fragments))[:num_max]:
            i = next(c)
            periods.append(yw_pop[i])

        # make cumulative weights over y-words that mimic power distribution
        logits = [(xi + 1) ** self.alpha for xi in range(len(periods))]
        cum_weights = [l / logits[-1] for l in logits]

        res = random.choices(periods, cum_weights=cum_weights, k=1000)  # simulate a distribution
        return res

    @cached_property
    def sim_mat_gold(self) -> np.ndarray:

        # every xi is related to every other xi, depending on num_fragments
        res = np.zeros((self.num_x, self.num_x))
        for row_id in range(self.num_x):
            offset = row_id % self.num_fragments
            res[row_id, offset::self.num_fragments] += 1

        return res
