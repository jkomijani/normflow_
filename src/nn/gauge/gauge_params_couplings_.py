# Copyright (c) 2021-2023 Javad Komijani

"""
This module contains new neural networks that are subclasses of `Module_`
and couple sites to each other.
"""


import torch
import numpy as np

from ..scalar.couplings_ import RQSplineCoupling_, MultiRQSplineCoupling_

pi = np.pi


class SU2RQSplineCoupling_(RQSplineCoupling_):
    """
    Special case of `RQSplineCoupling_` with following assumptions:

    1.  The input `x` is a phase between [0, 1],
        and the output will be in the same range.

    2.  The input `x` already has a channel axis,
        but we need to include cos and sin of the input.
    """

    def __init__(self, nets, xlim=(0, 1), ylim=(0, 1), **kwargs):

        super().__init__(nets, xlim=xlim, ylim=ylim, **kwargs)

    def preprocess_fz(self, x):  # fz: frozen
        return x

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return x


class U1RQSplineCoupling_(RQSplineCoupling_):
    """
    Special case of `RQSplineCoupling_` with following assumptions:

    1.  The input `x` is a phase between (-pi, pi],
        and the output will be in the same range.

    2.  The input `x` already has a channel axis,
        but we need to include cos and sin of the input.
    """

    def __init__(self, nets, xlim=(-pi, pi), ylim=(-pi, pi), **kwargs):

        super().__init__(nets, xlim=xlim, ylim=ylim, **kwargs)

    def preprocess_fz(self, x):  # fz: frozen
        # return x  # torch.cos(x)
        return torch.cat((torch.cos(x), torch.sin(x)), dim=self.channels_axis)

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return x


class SU3RQSplineCoupling_(MultiRQSplineCoupling_):
    r"""
    Special case of `RQSplineCoupling_` with following assumptions:

    boundaries : list[str]
        Possible values are ['none', 'none'], ['none', 'periodic']
        and ['periodic', 'periodic'].
        defines the boundary condition of the two input parameters.
        If 'none', the input parameter is scaled to [0, 1] and passed to the
        network as a single parameter. If 'periodic', the input parameter is
        scaled to [-pi, pi], split into a 2-tuple (cos(x), sin(x)) and passed
        to the network as two parameters.
    """

    def __init__(self, nets,
            xlims=[(0, 1), (-pi, pi)],
            ylims=[(0, 1), (-pi, pi)],
            boundaries=['none', 'periodic'], **kwargs
            ):

        super().__init__(nets, xlims=xlims, ylims=ylims, **kwargs)
        self.xlims = xlims
        self.ylims = ylims
        self.boundaries = boundaries

    def preprocess_fz(self, x):  # fz: frozen
        # split x into two parts, one for each independent parameter of the
        # eigenvectors of the SU(3) matrix
        # further split each part into cos and sin if boundary == 'periodic'
        bound0, bound1 = self.boundaries
        if bound0 == 'none' and bound1 == 'none':
            pass
        elif bound0 == 'none' and bound1 == 'periodic':
            # split x in two tensor views along index 1 of channels_axis
            x_split = torch.tensor_split(x, [1], dim=self.channels_axis)
            assert torch.equal(x_split[0], x.split([1, 1], dim=self.channels_axis)[0])
            assert torch.equal(x_split[1], x.split([1, 1], dim=self.channels_axis)[1])
            x = torch.cat(
                    (x_split[0], torch.cos(x_split[1]), torch.sin(x_split[1])),
                    dim=self.channels_axis
                    )
        else:
            # split x in two tensor views along index 1 of channels_axis
            x_split = torch.tensor_split(x, [1], dim=self.channels_axis)
            x = torch.cat(
                    (torch.cos(x_split[0]), torch.sin(x_split[0]),
                     torch.cos(x_split[1]), torch.sin(x_split[1])),
                    dim=self.channels_axis
                    )
        return x
