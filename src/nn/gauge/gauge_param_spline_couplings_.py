# Copyright (c) 2021-2023 Javad Komijani

"""
This module contains new neural networks that are subclasses of `Module_`
and couple sites to each other.
"""


import torch
import numpy as np

from ..scalar.couplings_ import RQSplineCoupling_, MultiRQSplineCoupling_


class SU2RQSplineCoupling_(RQSplineCoupling_):
    """Special case of `RQSplineCoupling_` but assuming the input  has already
    a channel axis.
    """
    def preprocess_fz(self, x):  # fz: frozen
        return x

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return x


class U1RQSplineCoupling_(RQSplineCoupling_):
    """Special case of `RQSplineCoupling_` but assuming the input  has already
    a channel axis, and we also include cos and sin of the input (times 2 *pi)
    """

    def preprocess_fz(self, x):  # fz: frozen
        x = x * (2*np.pi)
        return torch.cat((torch.cos(x), torch.sin(x)), dim=self.channels_axis)

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return x


class SU3RQSplineCoupling_(MultiRQSplineCoupling_):
    r"""
    Special case of `RQSplineCoupling_` with following assumptions:

    preprocess_fz_boundaries : list[str]
        Possible values are ['none', 'none'], ['none', 'periodic']
        and ['periodic', 'periodic'].
        defines the boundary condition of the two input parameters.
        If 'none', the input parameter is scaled to [0, 1] and passed to the
        network as a single parameter. If 'periodic', the input parameter is
        scaled to [-pi, pi], split into a 2-tuple (cos(x), sin(x)) and passed
        to the network as two parameters.
    """

    def __init__(self, nets,
            xlims=[(0, 1), (0, 1)],
            ylims=[(0, 1), (0, 1)],
            preprocess_fz_boundaries=['none', 'none'], **kwargs
            ):

        super().__init__(nets, xlims=xlims, ylims=ylims, **kwargs)
        self.preprocess_fz_boundaries = preprocess_fz_boundaries

    def preprocess_fz(self, x):  # fz: frozen
        # split x into two parts, one for each independent parameter of the
        # eigenvectors of the SU(3) matrix
        # further split each part into cos and sin if boundary == 'periodic'
        bound0, bound1 = self.preprocess_fz_boundaries
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
