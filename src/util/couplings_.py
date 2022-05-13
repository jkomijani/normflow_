# Copyright (c) 2021-2022 Javad Komijani

"""This module contains new neural networks that
1. are children of Module_
2. couple sites to each other

Note that as in Module_ itself, the trailing underscore implies that
the associate forward and `backward` methods handle the Jacobians of
the transformation.
"""


from .._normflowcore import np, torch, torch_device, Module_
from ..lib.spline import Pade22Spline
from .mask import Mask

import itertools


class CouplingBlock_(Module_):
    """Like a `torch.nn.Module` except for the `forward` and `backward` methods
    that handle the Jacobians of the transformation.
    We use trailing underscore to denote the neworks in which the `forward`
    and `backward` methods handle the Jacobians of the transformation.

    CouplingBlock_ contains two layers that uses masks to...
    """

    def __init__(self, net0, net1, equivariance=False,
            channels_axis=None, mask=None, label='coupling_', **mask_kwargs
            ):
        """If mask is not given, mask_shape &... should be provided."""
        super().__init__(label=label)
        self.net0, self.net1 = net0, net1
        self.mask = Mask(**mask_kwargs) if mask is None else mask
        self.equivariance = equivariance
        if channels_axis is not None:
            self.channels_axis = channels_axis

    # @torch.cuda.amp.autocast()
    def forward(self, x, log0=0):
        x_0, x_1 = self.mask.split(x)
        fnc = self.half_forward
        if self.net0 is not None:
            x_0, log0 = fnc(self.net0,
                            x_active=x_0, x_frozen=x_1, which_half=0, log0=log0
                            )
        if self.net1 is not None:
            x_1, log0 = fnc(self.net1,
                            x_active=x_1, x_frozen=x_0, which_half=1, log0=log0
                            )
        return self.mask.cat(x_0, x_1), log0

    def backward(self, x, log0=0):
        x_0, x_1 = self.mask.split(x)
        fnc = self.half_backward
        if self.net1 is not None:
            x_1, log0 = fnc(self.net1,
                            x_active=x_1, x_frozen=x_0, which_half=1, log0=log0
                            )
        if self.net0 is not None:
            x_0, log0 = fnc(self.net0,
                            x_active=x_0, x_frozen=x_1, which_half=0, log0=log0
                            )
        return self.mask.cat(x_0, x_1), log0

    def half_forward(self, net, *, x_active, x_frozen, which_half, log0=0):
        pass

    def half_backward(self, net, *, x_active, x_frozen, which_half, log0=0):
        pass

    def preprocess(self, x, normalize=False):
        if normalize:
            return x.unsqueeze(self.channels_axis) / (x.std() + 0.01)
        else:
            return x.unsqueeze(self.channels_axis)


class ShiftBlock_(CouplingBlock_):

    channels_axis = 1  # Default value

    def half_forward(self, net, *, x_active, x_frozen, which_half, log0=0):
        t = net(self.preprocess(x_frozen)).squeeze(self.channels_axis)
        return self.mask.purify(x_active + t, channel=which_half), log0

    def half_backward(self, net, *, x_active, x_frozen, which_half, log0=0):
        t = net(self.preprocess(x_frozen)).squeeze(self.channels_axis)
        return self.mask.purify(x_active - t, channel=which_half), log0

    def transfer(self, scale_factor=1, **mask_kwargs):
        """Do not forget to pass mask in **kwargs or pass the info for creating
        an appropriate mask.
        """
        # Todo: instead of passing mask, one should pass scale_factor and shape
        # to self.mask and transfer it to a new mask if mask_on_fly is
        # activated.
        return ShiftBlock_(
                self.net0.transfer(scale_factor=scale_factor),
                self.net1.transfer(scale_factor=scale_factor),
                equivariance=self.equivariance,
                channels_axis=self.channels_axis,
                **mask_kwargs
                )


class AffineBlock_(CouplingBlock_):

    channels_axis = 1  # Default value

    def half_forward(self, net, *, x_active, x_frozen, which_half, log0=0):
        out = net(self.preprocess(x_frozen))
        t, s = out.split((1, 1), dim=self.channels_axis)
        # purify: get rid of unwanted contributions to x_frozen
        t = self.mask.purify(t.squeeze(self.channels_axis), channel=which_half)
        s = self.mask.purify(s.squeeze(self.channels_axis), channel=which_half)
        if self.equivariance:
            s = torch.abs(s)
        return t + x_active * torch.exp(-s), log0 - self.sum_density(s)

    def half_backward(self, net, *, x_active, x_frozen, which_half, log0=0):
        out = net(self.preprocess(x_frozen))
        t, s = out.split((1, 1), dim=self.channels_axis)
        # purify: get rid of unwanted contributions to x_frozen
        t = self.mask.purify(t.squeeze(self.channels_axis), channel=which_half)
        s = self.mask.purify(s.squeeze(self.channels_axis), channel=which_half)
        if self.equivariance:
            s = torch.abs(s)
        return (x_active - t) * torch.exp(s), log0 + self.sum_density(s)

    def transfer(self, scale_factor=1, **mask_kwargs):
        """Do not forget to pass mask in **kwargs or pass the info for creating
        an appropriate mask.
        """
        # Todo: instead of passing mask, one should pass scale_factor and shape
        # to self.mask and transfer it to a new mask if mask_on_fly is
        # activated.
        return AffineBlock_(
                self.net0.transfer(scale_factor=scale_factor),
                self.net1.transfer(scale_factor=scale_factor),
                equivariance=self.equivariance,
                channels_axis=self.channels_axis,
                **mask_kwargs
                )


class Pade22SplineBlock_(CouplingBlock_):

    def __init__(self, net0, net1, width=2, channels_axis=-1, smooth=False,
            **kwargs
            ):
        super().__init__(net0, net1, **kwargs)
        self.width = width  # 2*B in paper
        self.channels_axis = channels_axis
        self.softmax = torch.nn.Softmax(dim=channels_axis)
        self.softplus = torch.nn.Softplus()
        self.smooth = smooth

    def half_forward(self, net, *, x_active, x_frozen, which_half, log0=0):
        out = net(self.preprocess(x_frozen))
        spline = self.make_spline(out)
        fx_active, g = spline(x_active, grad=True, squeezed=True)
        # g is the gradient of spline @ x_active
        fx_active = self.mask.purify(fx_active, channel=which_half)
        g = self.mask.purify(g, channel=which_half)
        return fx_active, log0 + self.sum_density(torch.log(g))

    def half_backward(self, net, *, x_active, x_frozen, which_half, log0=0):
        out = net(self.preprocess(x_frozen))
        spline = self.make_spline(out)
        fx_active, g = spline.backward(x_active, grad=True, squeezed=True)
        # g is the gradient of spline @ x_active
        fx_active = self.mask.purify(fx_active, channel=which_half)
        g = self.mask.purify(g, channel=which_half)
        return fx_active, log0 + self.sum_density(torch.log(g))

    def _hack(self, x, half=0):
        x_0, x_1 = self.mask.split(x)
        if half == 0:
            net, x_active, x_frozen = self.net0, x_0, x_1
        else:
            net, x_active, x_frozen = self.net1, x_1, x_0
        out = net(self.preprocess(x_frozen))
        spline = self.make_spline(out)
        return spline, x_active

    def make_spline(self, out):
        """construct a spline with number of knots deduced from input `out`
        and `self.smooth`.
        The first knot is always at `(-w/2, -w/2)` and the last knot is
        always at `(w/2, w/2)`, where `w = self.width`.
        The slope at the first and last knots is fixed to 1.
        Hence the number of channels in the input `out` should always be either
        `2 k` or `3 k - 1` depending on `self.smooth`.
        """
        # `out` is the output of net(in)
        axis = self.channels_axis
        def pad_like(x, const=0):
            shape = list(x.shape)
            shape[axis] = 1
            pad = torch.zeros if const == 0 else torch.ones
            return pad(tuple(shape))
        def to_coordinate(w):
            x = torch.cumsum(self.softmax(w), axis) * self.width
            return torch.cat((pad_like(x, const=0), x), axis) - self.width/2
        def to_derivatives(d):
            ones = pad_like(d, const=1)
            return torch.cat((ones, self.softplus(d), ones), axis)
        def map_(w, h, d=None):
            mydict = {'knots_x': to_coordinate(w), 'knots_y': to_coordinate(h)}
            mydict['knots_d'] = None if d is None else to_derivatives(d)
            return mydict
        n = out.shape[axis]
        m = n//2 if self.smooth else (n+1)//3
        sizes = (m, m) if self.smooth else (m, m, n - 2*m)
        kwargs = map_(*out.split(sizes, dim=axis))
        kwargs['knots_axis'] = axis
        kwargs.update(dict(extrap_left='linear', extrap_right='linear'))
        return Pade22Spline(**kwargs)


P22SBlock_ = Pade22SplineBlock_  # alias
RQSBlock_ = Pade22SplineBlock_  # alias
