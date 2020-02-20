from cached_property import cached_property
import random
from itertools import cycle
import numpy as np

from straddler.outcomes import get_outcomes


class ToyCorpus:
    """
    methods for making a document, a string of artificial words following the structure (xw, yw, xw, yw, ..).
    example document: "x1 y5 x34 y82 x93 y3 x45 y11".
    """

    def __init__(self,
                 doc_size: int = 100_000,
                 num_types: int = 4096,
                 num_xws: int = 512,
                 num_fragments: int = 2,  # number of sub-categories in xws
                 fragmentation_prob: float = 0.5,
                 ) -> None:
        self.doc_size = doc_size
        self.num_types = num_types
        self.num_xws = num_xws
        self.num_fragments = num_fragments
        self.fragmentation_prob = fragmentation_prob
        self.num_yws = self.num_types - self.num_xws

        self.xws = [f'x{i:0>6}' for i in range(self.num_xws - 1)]  # leave space for straddler
        self.yws = [f'y{i:0>6}' for i in range(self.num_yws - 0)]

        # straddler
        self.straddler = f's{len(self.xws):0>6}'
        assert self.straddler not in self.xws
        self.xws.append(self.straddler)

        # a smaller set of yws
        self.yws_limited = self.yws[::num_fragments]

        # map subsets of xws to mutually exclusive subsets of yws
        c = cycle([self.yws[offset::num_fragments] for offset in range(num_fragments)])
        self.xw2yws = {xw: next(c) for xw in self.xws}
        self.xw2yws[self.straddler] = self.yws  # straddler co-occurs with ALL y-words

        print('Initialized ToyCorpus')
        print(f'Lowest theoretical pp ={len(self.yws_limited):>6,}')
        print(f'Number of limited yws ={len(self.yws_limited):>6,}')
        print(f'Number of y-word types={self.num_yws:>6,}')

    @cached_property
    def doc(self) -> str:
        res = ''
        for n in range(self.doc_size // 2):  # divide by 2 because each loop adds 2 words

            # sample xw randomly
            xw = random.choice(self.xws)

            # sample yw from one of many sub-population (this keeps overall type frequency the same)
            if random.random() < self.fragmentation_prob:
                yw = random.choice(self.xw2yws[xw])
            # sample yw from a single sub-population
            else:
                yw = random.choice(self.yws_limited)
            res += f'{xw} {yw} '  # whitespace after each
        return res

    @cached_property
    def sim_mat_gold(self) -> np.ndarray:

        # every xw is related to every other xw, depending on num_fragments
        res = np.zeros((self.num_xws, self.num_xws))
        for row_id in range(self.num_xws):
            offset = row_id % self.num_fragments
            res[row_id, offset::self.num_fragments] += 1

        # last row is for straddler
        res[-1] = [1] * self.num_xws

        return res