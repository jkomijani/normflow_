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

    def preprocess_fz(self, x):  # fz: frozen
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
       but we need to include cos and sin of the input.
    """

    def __init__(self, net0, net1, xlim=(-pi, pi), ylim=(-pi, pi), **kwargs):

        super().__init__(net0, net1, xlim=xlim, ylim=ylim, **kwargs)

    def preprocess_fz(self, x):  # fz: frozen
        # return x  # torch.cos(x)
        return torch.cat((torch.cos(x), torch.sin(x)), dim=self.channels_axis)

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return x


class SU3RQSplineBlock_(RQSplineBlock_):
    r"""Special case of `RQSplineBlock_` with following assumptions:
    1. The input `x` is a phase between [0, 1],
       and the output will be in the same range.
    2. The input `x` already has a channel axis,
       but we need to include cosine and since of the input.
    boundary : str \in ['none' or 'periodic']
        defines the boundary condition of the input parameter.
        If 'none', the input parameter is scaled to [0, 1] and passed to the
        network as a single parameter. If 'periodic', the input parameter is
        scaled to [-pi, pi], split into a 2-tuple (cos(x), sin(x)) and passed
        to the network as two parameters.
    """

    def __init__(self, net0, net1,
            xlims=[(0, 1), (-pi, pi)],
            ylims=[(0, 1), (-pi, pi)],
            boundaries=['none', 'periodic'], **kwargs
            ):

        super().__init__(net0, net1, xlim=(0, 1), ylim=(0, 1), **kwargs)
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
            x_split = list(x.split([1, 1], dim=self.channels_axis))
            x = torch.cat(
                    (x_split[0], torch.cos(x_split[1]), torch.sin(x_split[1])),
                    dim=self.channels_axis
                    )
        else:
            x_split = list(x.split([1, 1], dim=self.channels_axis))
            x = torch.cat(
                    (torch.cos(x_split[0]), torch.sin(x_split[0]),
                     torch.cos(x_split[1]), torch.sin(x_split[1])),
                    dim=self.channels_axis
                    )
        return x
    
    def preprocess(self, x):
        # split x into two parts, one for each independent parameter of the eigenvectors of the SU(3) matrix
        xs = list(x.split([1, 1], dim=self.channels_axis))
        return xs

    def postprocess(self, xs):
        # concatenate list of x_active channels into single tensor
        x = torch.cat(xs, dim=self.channels_axis)
        return x
    
    def apply_spline(self, x_actives, splines, backward=False):
        gs = [None, None]
        for i in range(2):
            if self.xlims[i] != (0, 1):
                # affinely scale x_actives[i] to [0, 1]
                x_actives[i] = x_actives[i] - self.xlims[i][0]
                x_actives[i] = x_actives[i] / (self.xlims[i][1] - self.xlims[i][0])
            # apply backward or forward spline transformation
            if backward:
                x_actives[i], gs[i] = splines[i].backward(x_actives[i], grad=True)
            else:
                x_actives[i], gs[i] = splines[i](x_actives[i], grad=True)
            if self.ylims[i] != (0, 1):
                # affinely scale x_actives[i] back to self.ylims[i]
                x_actives[i] = x_actives[i] * (self.ylims[i][1] - self.ylims[i][0])
                x_actives[i] = x_actives[i] + self.ylims[i][0]
        return x_actives, gs

    def half_forward(self, net, *, x_active, x_frozen, which_half, log0=0):
        out = net(self.preprocess_fz(x_frozen))
        spline = self.make_spline(out)
        # below g is the gradient of spline @ x_active
        fx_active, g = self.apply_spline(self.preprocess(x_active), spline)
        fx_active, g = self.postprocess(fx_active), self.postprocess(g)
        fx_active = self.mask.purify(fx_active, channel=which_half)
        g = self.mask.purify(g, channel=which_half, zero2one=True)
        return fx_active, log0 + self.sum_density(torch.log(g))
    
    def half_backward(self, net, *, x_active, x_frozen, which_half, log0=0):
        out = net(self.preprocess_fz(x_frozen))
        spline = self.make_spline(out)
        # below g is the gradient of spline @ x_active
        fx_active, g = self.apply_spline(self.preprocess(x_active), spline, backward=True)
        fx_active, g = self.postprocess(fx_active), self.postprocess(g)
        fx_active = self.mask.purify(fx_active, channel=which_half)
        g = self.mask.purify(g, channel=which_half, zero2one=True)
        return fx_active, log0 + self.sum_density(torch.log(g))

    def make_spline(self, out):
        """splits the out in self.channel_axis into two equal parts and makes two splines,
        one for each independent parameter of the eigenvectors of the SU(3) matrix.
        """
        out_splits = list(out.split([13, 13], dim=self.channels_axis))
        splines = [super(SU3RQSplineBlock_, self).make_spline(out_split) for out_split in out_splits]
        return splines
