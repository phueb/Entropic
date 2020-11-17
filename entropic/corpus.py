from cached_property import cached_property
import random
from itertools import cycle
import numpy as np
from typing import Tuple, Optional, List


class Corpus:
    """
    methods for making a document, a string of artificial words following the structure (A, X, B, Y).

    X elements always predict a subset of Y elements, in every sequence.
    redundancy argument influences how likely an X element predicts an A or B element.

    there are 3 redundancy reduction strategies:
    1) lowering the probability of redundancy directly
    2) reducing the size of category A or B
    3) dropping slots A or B in some sequences
    """

    def __init__(self,
                 doc_size: int,
                 num_types: int,
                 num_fragments: int,
                 starvation: Tuple[float, float],
                 redundant_a: Tuple[float, float],
                 redundant_b: Tuple[float, float],
                 size_a: Tuple[float, float],
                 size_b: Tuple[float, float],
                 drop_a: Tuple[float, float],
                 drop_b: Tuple[float, float],
                 alpha: float = 2.0,
                 seed: Optional[int] = None,
                 ) -> None:

        assert 0.0 <= redundant_a[0] <= 1.0
        assert 0.0 <= redundant_a[1] <= 1.0
        assert 0.0 <= redundant_b[0] <= 1.0
        assert 0.0 <= redundant_b[1] <= 1.0

        self.doc_size = doc_size
        self.num_types = num_types
        self.num_fragments = num_fragments
        self.starvation = starvation
        self.redundant_a = redundant_a
        self.redundant_b = redundant_b
        self.size_a = size_a
        self.size_b = size_b
        self.drop_a = drop_a
        self.drop_b = drop_b
        self.alpha = alpha

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

        # map xi to category-relevant yi
        y_fragments = [self.y[offset::num_fragments] for offset in range(num_fragments)]
        self.xi2y = {xi: y_fragments[self.xi2cat_id[xi]] for xi in self.x}

        # make each xi redundant with one ai, bi
        self.xi2ai = {xi: ai for xi, ai in zip(self.x, self.a)}
        self.xi2bi = {xi: bi for xi, bi in zip(self.x, self.b)}

        if seed is not None:
            random.seed(seed)

    @cached_property
    def sequences(self) -> str:

        doc_size1 = self.doc_size
        doc_size2 = self.doc_size
        nw1 = doc_size1 // self.num_words_in_window
        nw2 = doc_size2 // self.num_words_in_window

        ra1, ra2 = self.redundant_a
        rb1, rb2 = self.redundant_b
        iai1 = iter(np.linspace(ra1, np.mean([ra1, ra2]), nw1))
        iai2 = iter(np.linspace(np.mean([ra1, ra2]), ra2, nw2))
        ibi1 = iter(np.linspace(rb1, np.mean([rb1, rb2]), nw1))
        ibi2 = iter(np.linspace(np.mean([rb1, rb2]), rb2, nw2))

        sa1, sa2 = self.size_a
        sb1, sb2 = self.size_b
        sai1 = iter(np.linspace(sa1, np.mean([sa1, sa2]), nw1))
        sai2 = iter(np.linspace(np.mean([sa1, sa2]), sa2, nw2))
        sbi1 = iter(np.linspace(sb1, np.mean([sb1, sb2]), nw1))
        sbi2 = iter(np.linspace(np.mean([sb1, sb2]), sb2, nw2))

        da1, da2 = self.drop_a
        db1, db2 = self.drop_b
        dai1 = iter(np.linspace(da1, np.mean([da1, da2]), nw1))
        dai2 = iter(np.linspace(np.mean([da1, da2]), da2, nw2))
        dbi1 = iter(np.linspace(db1, np.mean([db1, db2]), nw1))
        dbi2 = iter(np.linspace(np.mean([db1, db2]), db2, nw2))

        doc1 = self.make_doc(nw1, self.starvation[0], iai1, ibi1, sai1, sbi1, dai1, dbi1)
        doc2 = self.make_doc(nw2, self.starvation[1], iai2, ibi2, sai2, sbi2, dai2, dbi2)
        return doc1 + doc2

    def make_doc(self,
                 num_windows: int,
                 starvation: float,
                 redundant_a: iter,
                 redundant_b: iter,
                 size_a: iter,
                 size_b: iter,
                 a_drop: iter,
                 b_drop: iter,
                 ) -> str:

        res = ''
        for n in range(num_windows):

            # sample xi
            xi = random.choice(self.x)  # do not sample from itertools.cycle because of predictable structure

            # read next in iterators
            rpa = next(redundant_a)
            rpb = next(redundant_b)
            sa = next(size_a)
            sb = next(size_b)
            da = next(a_drop)
            db = next(b_drop)

            # sample ai, bi from all possible ai, bi
            ai = random.choice(self.a[:int(sa * self.num_a)])
            bi = random.choice(self.b[:int(sb * self.num_b)])

            # chose redundant ai, bi
            if random.random() < rpa:
                ai = self.xi2ai[xi]
            if random.random() < rpb:
                bi = self.xi2bi[xi]

            # sample yi
            if random.random() < starvation:
                yi = random.choice(self.random_periods)
            else:
                yi = random.choice(self.xi2y[xi])

            # collect
            if random.random() < da and random.random() < db:
                res += f'{xi} {yi} '
            elif random.random() < da:
                res += f'{xi} {bi} {yi} '
            elif random.random() < db:
                res += f'{ai} {xi} {yi} '
            else:
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
