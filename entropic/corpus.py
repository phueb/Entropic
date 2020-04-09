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
                 incongruent_a: Tuple[float, float],
                 incongruent_b: Tuple[float, float],
                 size_a: Tuple[float, float],
                 size_b: Tuple[float, float],
                 alpha: float = 2.0,
                 seed: Optional[int] = None,
                 ) -> None:

        assert 0.0 <= incongruent_a[0] <= 1.0
        assert 0.0 <= incongruent_a[1] <= 1.0
        assert 0.0 <= incongruent_b[0] <= 1.0
        assert 0.0 <= incongruent_b[1] <= 1.0

        self.doc_size = doc_size
        self.num_types = num_types
        self.num_fragments = num_fragments
        self.starvation = starvation
        self.sample_b = sample_b
        self.sample_a = sample_a
        self.incongruent_a = incongruent_a
        self.incongruent_b = incongruent_b
        self.size_a = size_a
        self.size_b = size_b
        self.alpha = alpha
        self.delay = delay
        self.num_sentinels = num_sentinels

        self.incongruent_a = incongruent_a
        self.incongruent_b = incongruent_b

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

        if seed is not None:
            random.seed(seed)

    @cached_property
    def doc(self) -> str:

        doc_size1 = self.delay
        doc_size2 = self.doc_size - self.delay
        nw1 = doc_size1 // self.num_words_in_window
        nw2 = doc_size2 // self.num_words_in_window

        ia1, ia2 = self.incongruent_a
        ib1, ib2 = self.incongruent_b
        iai1 = iter(np.linspace(ia1, np.mean([ia1, ia2]), nw1))
        iai2 = iter(np.linspace(np.mean([ia1, ia2]), ia2, nw2))
        ibi1 = iter(np.linspace(ib1, np.mean([ib1, ib2]), nw1))
        ibi2 = iter(np.linspace(np.mean([ib1, ib2]), ib2, nw2))

        sa1, sa2 = self.size_a
        sb1, sb2 = self.size_b
        sai1 = iter(np.linspace(sa1, np.mean([sa1, sa2]), nw1))
        sai2 = iter(np.linspace(np.mean([sa1, sa2]), sa2, nw2))
        sbi1 = iter(np.linspace(sb1, np.mean([sb1, sb2]), nw1))
        sbi2 = iter(np.linspace(np.mean([sb1, sb2]), sb2, nw2))

        doc1 = self.make_doc(self.x1, nw1, self.starvation[0], self.sample_a[0], self.sample_b[0], iai1, ibi1, sai1, sbi1)
        doc2 = self.make_doc(self.x2, nw2, self.starvation[1], self.sample_a[1], self.sample_b[1], iai2, ibi2, sai2, sbi2)
        return doc1 + doc2

    def make_doc(self,
                 x: List[str],
                 num_windows: int,
                 starvation: float,
                 sample_a: str,
                 sample_b: str,
                 incongruent_a: iter,
                 incongruent_b: iter,
                 size_a: iter,
                 size_b: iter,
                 ) -> str:

        res = ''
        for n in range(num_windows):

            # sample xi
            xi = random.choice(x)  # do not sample from itertools.cycle because of predictable structure

            # change set size of a and b
            sa = next(size_a)
            sb = next(size_b)

            # sample ai
            if sample_a == 'item':
                ai = self.xi2ai[xi]
            elif sample_a == 'sub':
                ai = random.choice(self.xi2a[xi])
            elif sample_a == 'super':
                ai = random.choice(self.a[:int(sa * self.num_a)])
            else:
                raise AttributeError('Invalid arg to "sample_a".')

            # sample bi
            if sample_b == 'item':
                bi = self.xi2bi[xi]
            elif sample_b == 'sub':
                bi = random.choice(self.xi2b[xi])
            elif sample_b == 'super':
                bi = random.choice(self.b[:int(sb * self.num_b)])
            else:
                raise AttributeError('Invalid arg to "sample_b".')

            # incongruent ai
            ipa = next(incongruent_a)
            if random.random() < ipa:
                ai = random.choice([ai for ai in self.a if ai not in self.xi2a[xi]])

            # incongruent bi
            ipb = next(incongruent_b)
            if random.random() < ipb:
                bi = random.choice([bi for bi in self.b if bi not in self.xi2b[xi]])

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
