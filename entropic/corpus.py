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
                 starvation: Tuple[Tuple[float], Tuple[float]],
                 redundant_a: Tuple[Tuple[float], Tuple[float]],
                 redundant_b: Tuple[Tuple[float], Tuple[float]],
                 size_a: Tuple[Tuple[float], Tuple[float]],
                 size_b: Tuple[Tuple[float], Tuple[float]],
                 drop_a: Tuple[Tuple[float], Tuple[float]],
                 drop_b: Tuple[Tuple[float], Tuple[float]],
                 alpha: float = 2.0,
                 seed: Optional[int] = None,
                 ) -> None:
        """

        :param doc_size:
        :param num_types:
        :param num_fragments:
        :param starvation:  probability of replacing y-word with category-irrelevant word (e.g. period)
        :param redundant_a:
        :param redundant_b:
        :param size_a: proportion of set size of a-words available for sampling
        :param size_b: proportion of set size of b-words available for sampling
        :param drop_a:
        :param drop_b:
        :param alpha:
        :param seed:
        """

        for i in range(2):
            for j in range(2):
                assert 0.0 <= starvation[i][j] <= 1.0
                assert 0.0 <= redundant_a[i][j] <= 1.0
                assert 0.0 <= redundant_b[i][j] <= 1.0
                assert 0.0 <= size_a[i][j] <= 1.0
                assert 0.0 <= size_b[i][j] <= 1.0
                assert 0.0 <= drop_a[i][j] <= 1.0
                assert 0.0 <= drop_b[i][j] <= 1.0


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

        nw = self.doc_size // self.num_words_in_window

        docs = ''
        for doc_id in range(2):

            starvation_i = iter(np.linspace(*self.starvation[doc_id], nw))

            rai = iter(np.linspace(*self.redundant_a[doc_id], nw))
            rbi = iter(np.linspace(*self.redundant_b[doc_id], nw))

            sai = iter(np.linspace(*self.size_a[doc_id], nw))
            sbi = iter(np.linspace(*self.size_b[doc_id], nw))

            dai = iter(np.linspace(*self.drop_a[doc_id], nw))
            dbi = iter(np.linspace(*self.drop_b[doc_id], nw))

            docs += self.make_doc(nw, starvation_i, rai, rbi, sai, sbi, dai, dbi)

        return docs

    def make_doc(self,
                 num_windows: int,
                 starvation: iter,
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
            starve = next(starvation)
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
            if random.random() < starve:
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
