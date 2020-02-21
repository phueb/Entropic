from cached_property import cached_property
import random
from itertools import cycle
import numpy as np


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

        self.xws = [f'x{i:0>6}' for i in range(self.num_xws)]
        self.yws = [f'y{i:0>6}' for i in range(self.num_yws)]

        # map subsets of xws to mutually exclusive subsets/fragments of yws
        yw_fragments = [self.yws[offset::num_fragments] for offset in range(num_fragments)]
        c = cycle(yw_fragments)
        self.xw2yws = {xw: next(c) for xw in self.xws}

        # a smaller set of yws - that has an equal amount of yws from each category, giving rise to a third category
        # sampling from these gives rise to third category which should be equally different from other categories
        self.yws_extra_fragment = []
        c = cycle(range(num_fragments))
        for yw_pop in zip(*yw_fragments):
            i = next(c)
            self.yws_extra_fragment.append(yw_pop[i])

        num_shared_with_fragment1 = len(set(self.yws_extra_fragment).intersection(yw_fragments[0]))
        num_shared_with_fragment2 = len(set(self.yws_extra_fragment).intersection(yw_fragments[1]))
        assert num_shared_with_fragment1 == num_shared_with_fragment2

        print('Initialized ToyCorpus')
        print(f'Lowest theoretical pp ={len(self.yws_extra_fragment):>6,}')
        print(f'Number of limited yws ={len(self.yws_extra_fragment):>6,}')
        print(f'Number of y-word types={self.num_yws:>6,}')

        assert len(self.yws_extra_fragment) == self.num_yws // num_fragments

    @cached_property
    def doc(self) -> str:
        res = ''
        for n in range(self.doc_size // 2):  # divide by 2 because each loop adds 2 words

            # sample xw randomly
            xw = random.choice(self.xws)

            # sample yw from one of many subset
            if random.random() < self.fragmentation_prob or xw in self.xws[:2]:  # first two x-words should be pure
                yw = random.choice(self.xw2yws[xw])
            # sample yw from a single subset that is equally different from all other subsets
            else:
                yw = random.choice(self.yws_extra_fragment)  # TODO select from limited sub pop unique to each xw
            res += f'{xw} {yw} '  # whitespace after each
        return res

    @cached_property
    def sim_mat_gold(self) -> np.ndarray:

        # every xw is related to every other xw, depending on num_fragments
        res = np.zeros((self.num_xws, self.num_xws))
        for row_id in range(self.num_xws):
            offset = row_id % self.num_fragments
            res[row_id, offset::self.num_fragments] += 1

        return res