from typing import List
from cached_property import cached_property
import random
from itertools import cycle


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

        # a smaller set of yws
        self.yws_limited = [o for o in self.yws if float(o[1:]) % num_fragments == 0]

        # subsets of xws precede subsets of yws
        c = cycle([self.yws[offset::num_fragments] for offset in range(num_fragments)])
        self.xw2yws = {xw: next(c) for xw in self.xws}

        for k, v in self.xw2yws.items():
            print(k)
            print(v)

        print('Initialized ToyCorpus with number of limited yws:', len(self.yws_limited))

    @cached_property
    def doc(self) -> str:
        res = ''
        for n in range(self.doc_size):

            # sample xw randomly
            xw = random.choice(self.xws)

            # sample next-word from one of many sub-population (this keeps overall type frequency the same)
            if random.random() < self.fragmentation_prob:
                yw = random.choice(self.xw2yws[xw])
            # sample next-word from a single sub-population
            else:
                yw = random.choice(self.yws_limited)
            res += f'{xw} {yw} '  # whitespace after each
        return res