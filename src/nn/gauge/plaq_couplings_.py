# Copyright (c) 2021-2022 Javad Komijani

"""This module contains new neural networks that
1. are children of Module_,
2. couple sites to each other.

Note that as in `Module_` itself, the trailing underscore implies that
the associate `forward` and `backward` methods handle the Jacobians of
the transformation.
"""


import torch
import numpy as np

from ..scalar.couplings_ import RQSplineBlock_

pi = np.pi


class SU2RQSplineBlock_(RQSplineBlock_):
    """Special case of `RQSplineBlock_` with following assumptions:
    1. The input `x` is a phase between [0, 1],
       and the output will be in the same range.
    2. The input `x` already has a channel axis,
       but we need to include cosine and since of the input.
    """

    def __init__(self, net0, net1, xlim=(0, 1), ylim=(0, 1), **kwargs):

        super().__init__(net0, net1, xlim=xlim, ylim=ylim, **kwargs)

    def preprocess_fz(self, x):
        return x

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return x


class U1RQSplineBlock_(RQSplineBlock_):
    """Special case of `RQSplineBlock_` with following assumptions:
    1. The input `x` is a phase between (-pi, pi],
       and the output will be in the same range.
    2. The input `x` already has a channel axis,
       but we need to include cosine and since of the input.
    """

    def __init__(self, net0, net1, xlim=(-pi, pi), ylim=(-pi, pi), **kwargs):

        super().__init__(net0, net1, xlim=xlim, ylim=ylim, **kwargs)

    def preprocess_fz(self, x):
        # return x  # torch.cos(x)
        return torch.cat((torch.cos(x), torch.sin(x)), dim=self.channels_axis)

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return x
