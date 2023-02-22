# Copyright (c) 2021-2022 Javad Komijani

"""This module contains new neural networks that
1. are children of Module_,
2. couple sites to each other.

Note that as in Module_ itself, the trailing underscore implies that
the associated forward and backward methods handle the Jacobians of
the transformation.
"""


import torch
import numpy as np

from .._core import Module_
from ...lib.spline import RQSpline


class CouplingBlock_(Module_):
    """A template class for a block of invertible transformations using a
    mask-based approach to divide the input into two partitions that are coupled
    in a specif way that makes it easy to calculate the Jacobian of the
    transformation.

    The block conains two layers and takes as input two NNs. The first (second)
    NN is used in the first (second) layer to transform the first (second)
    partition, while the second (first) patition is kept fixed.

    Three examples of children classes are ShiftBlock_, AffineBlock_, and
    RQSplineBlock_.

    Parameters
    ----------
    net0 : an instance of torch.nn
         The output of net0 must have two output channels for ShiftBlock_ and
         AffineBlock_ and enough channels for RQSplineBlock_.

    net1 : an instance of torch.nn
         Similar to net0 but for the second layer of the block.

    mask : a tensor one 0s and 1s
         For partitioning the input.

    channels_axis : int, default 1
        The channel axis in the outputs of net0 and net1.

    zee2sym : bool
        If True explicitely implements the Z_2 symmetery.

    label : str
        Can be used for unique labeling of NNs.
    """

    def __init__(self, net0, net1, *, mask,
            zee2sym=False, channels_axis=1, label='coupling_'
            ):
        super().__init__(label=label)
        self.net0, self.net1 = net0, net1
        self.mask = mask
        self.zee2sym = zee2sym
        self.channels_axis = channels_axis

    # @torch.cuda.amp.autocast()
    def forward(self, x, log0=0):
        x_0, x_1 = self.mask.split(x)
        if self.net0 is not None:
            x_0, log0 = self.half_forward(self.net0,
                            x_active=x_0, x_frozen=x_1, which_half=0, log0=log0
                            )
        if self.net1 is not None:
            x_1, log0 = self.half_forward(self.net1,
                            x_active=x_1, x_frozen=x_0, which_half=1, log0=log0
                            )
        return self.mask.cat(x_0, x_1), log0

    def backward(self, x, log0=0):
        x_0, x_1 = self.mask.split(x)
        if self.net1 is not None:
            x_1, log0 = self.half_backward(self.net1,
                            x_active=x_1, x_frozen=x_0, which_half=1, log0=log0
                            )
        if self.net0 is not None:
            x_0, log0 = self.half_backward(self.net0,
                            x_active=x_0, x_frozen=x_1, which_half=0, log0=log0
                            )
        return self.mask.cat(x_0, x_1), log0

    def half_forward(self, net, *, x_active, x_frozen, which_half, log0=0):
        pass

    def half_backward(self, net, *, x_active, x_frozen, which_half, log0=0):
        pass

    def preprocess_fz(self, x):  # fz: frozen
        return x.unsqueeze(self.channels_axis)

    def preprocess(self, x):
        return x.unsqueeze(self.channels_axis)

    def postprocess(self, x):
        return x.squeeze(self.channels_axis)

    def transfer(self, scale_factor=1, mask=None, **extra):
        return self.__class__(
                self.net0.transfer(scale_factor=scale_factor),
                self.net1.transfer(scale_factor=scale_factor),
                mask=self.mask if mask is None else mask,
                label=self.label,
                zee2sym=self.zee2sym,
                channels_axis=self.channels_axis
                )


class ShiftBlock_(CouplingBlock_):
    """A coupling block with a shift transformation."""

    def half_forward(self, net, *, x_active, x_frozen, which_half, log0=0):
        t = self.postprocess(net(self.preprocess_fz(x_frozen)))
        return self.mask.purify(x_active + t, channel=which_half), log0

    def half_backward(self, net, *, x_active, x_frozen, which_half, log0=0):
        t = self.postprocess(net(self.preprocess_fz(x_frozen)))
        return self.mask.purify(x_active - t, channel=which_half), log0


class AffineBlock_(CouplingBlock_):
    """A coupling block with an affine transformation."""

    def half_forward(self, net, *, x_active, x_frozen, which_half, log0=0):
        out = net(self.preprocess_fz(x_frozen))
        t, s = out.split((1, 1), dim=self.channels_axis)
        # purify: get rid of unwanted contributions to x_frozen
        t = self.mask.purify(self.postprocess(t), channel=which_half)
        s = self.mask.purify(self.postprocess(s), channel=which_half)
        if self.zee2sym:
            s = torch.abs(s)
        return t + x_active * torch.exp(-s), log0 - self.sum_density(s)

    def half_backward(self, net, *, x_active, x_frozen, which_half, log0=0):
        out = net(self.preprocess_fz(x_frozen))
        t, s = out.split((1, 1), dim=self.channels_axis)
        # purify: get rid of unwanted contributions to x_frozen
        t = self.mask.purify(self.postprocess(t), channel=which_half)
        s = self.mask.purify(self.postprocess(s), channel=which_half)
        if self.zee2sym:
            s = torch.abs(s)
        return (x_active - t) * torch.exp(s), log0 + self.sum_density(s)


class RQSplineBlock_(CouplingBlock_):
    """A coupling block with a rational quadratic spline transformation."""

    def __init__(self, net0, net1, *, mask,
            xlim=(0, 1), ylim=(0, 1), knots_x=None, knots_y=None, extrap={},
            channels_axis=1, label='spline_coupling_'
            ):
        """
        Tips on extrapolation:
        1.  for linear extrapolation on both sides set
            `extrap=dict(left='linear', right='linear')`
        2.  for linear extrapolation on right and anti-periodic boundary on left
            set `extrap={'left': 'anti', 'right': 'linear'}`.
        """

        super().__init__(net0, net1,
                mask=mask, channels_axis=channels_axis, label=label
                )

        self.xlim, self.xwidth = xlim, xlim[1] - xlim[0]
        self.ylim, self.ywidth = ylim, ylim[1] - ylim[0]
        self.knots_x = knots_x
        self.knots_y = knots_y
        self.extrap = extrap

        self.softmax = torch.nn.Softmax(dim=self.channels_axis)
        self.softplus = torch.nn.Softplus(beta=np.log(2))
        # we set the beta of Softplus to log(2) so that self.softplust(0)
        # returns 1. With this setting it would be easy to set the derivatives
        # to 1 (with zero inputs).

    def half_forward(self, net, *, x_active, x_frozen, which_half, log0=0):
        out = net(self.preprocess_fz(x_frozen))
        spline = self.make_spline(out)
        # below g is the gradient of spline @ x_active
        fx_active, g = spline(self.preprocess(x_active), grad=True)
        fx_active, g = self.postprocess(fx_active), self.postprocess(g)
        # the above two lines are equivalent to the following for default cases
        # fx_active, g = spline(x_active, grad=True, squeezed=True)
        fx_active = self.mask.purify(fx_active, channel=which_half)
        g = self.mask.purify(g, channel=which_half, zero2one=True)
        return fx_active, log0 + self.sum_density(torch.log(g))

    def half_backward(self, net, *, x_active, x_frozen, which_half, log0=0):
        out = net(self.preprocess_fz(x_frozen))
        spline = self.make_spline(out)
        # below g is the gradient of spline @ x_active
        fx_active, g = spline.backward(self.preprocess(x_active), grad=True)
        fx_active, g = self.postprocess(fx_active), self.postprocess(g)
        # the above two lines are equivalent to the following for default cases
        # fx_active, g = spline.backward(x_active, grad=True, squeezed=True)
        fx_active = self.mask.purify(fx_active, channel=which_half)
        g = self.mask.purify(g, channel=which_half, zero2one=True)
        return fx_active, log0 + self.sum_density(torch.log(g))

    def _hack(self, x, which_half=0):
        x_0, x_1 = self.mask.split(x)
        if which_half == 0:
            net, x_active, x_frozen = self.net0, x_0, x_1
        else:
            net, x_active, x_frozen = self.net1, x_1, x_0
        out = net(self.preprocess_fz(x_frozen))
        spline = self.make_spline(out)
        return spline, x_active

    def make_spline(self, out):
        # `out` is the output of net(in)
        """construct a spline with number of knots deduced from input `out`.
        The first knot is always at `(xlim[0], ylim[0])` and the last knot is
        always at `(xlim[1], ylim[1])`; hence, the number of channels in the
        input `out` should always be `3 m - 2` unless one fixes knots_x or
        knots_y. Here, `m` is the number of knots in the spline.

        To clarify more, the input `out` gets split into (m-1, m-1, m) parts
        corresponding to knots_x, knots_y, and knots_d.
        When either knots_x or knots_y is already fixed, the input `out` gets
        split into (m-1, m) parts and if both are fixed ther will be no
        partitioning.
        """

        axis = self.channels_axis
        knots_x = self.knots_x
        knots_y = self.knots_y

        def zeropad(w):
            pad_shape = list(w.shape)
            pad_shape[axis] = 1  # note that axis migh be e.g. -1
            return torch.zeros(pad_shape)

        cumsumsoftmax = lambda w: torch.cumsum(self.softmax(w), dim=axis)
        to_coord = lambda w: torch.cat((zeropad(w), cumsumsoftmax(w)), dim=axis)
        to_deriv = lambda d: self.softplus(d) if d is not None else None

        n = out.shape[axis]  # n parameters to specify splines
        if knots_x is None and knots_y is None:
            m = (n + 2) // 3
            x_, y_, d_ = out.split((m-1, m-1, m), dim=axis)
            knots_x = to_coord(x_) * self.xwidth + self.xlim[0]
            knots_y = to_coord(y_) * self.ywidth + self.ylim[0]
            knots_d = to_deriv(d_)
        elif knots_x is not None and knots_y is None:
            m = (n + 2) // 2
            y_, d_ = out.split((m-1, m), dim=axis)
            knots_y = to_coord(y_) * self.ywidth + self.ylim[0]
            knots_d = to_deriv(d_)
        elif knots_x is None and knots_y is not None:
            m = (n + 2) // 2
            x_, d_ = out.split((m-1, m), dim=axis)
            knots_x = to_coord(x_) * self.xwidth + self.xlim[0]
            knots_d = to_deriv(d_)
        else:
            knots_d = to_deriv(out)

        kwargs = dict(knots_x=knots_x, knots_y=knots_y, knots_d=knots_d)
        kwargs.update(dict(knots_axis=axis, extrap=self.extrap))

        return RQSpline(**kwargs)

    def transfer(self, scale_factor=1, mask=None, **extra):
        return self.__class__(
                self.net0.transfer(scale_factor=scale_factor),
                self.net1.transfer(scale_factor=scale_factor),
                mask=self.mask if mask is None else mask,
                label=self.label,
                zee2sym=self.zee2sym,
                channels_axis=self.channels_axis,
                xlim=self.xlim,
                ylim=self.ylim,
                knots_x=self.knots_x,
                knots_y=self.knots_y,
                extrap=self.extrap
                )
