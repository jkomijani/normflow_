# Copyright (c) 2021-2023 Javad Komijani

"""
This module contains new neural networks that are subclasses of Module_ and
couple sites to each other.

As in Module_, the trailing underscore implies that the associated forward and
backward methods handle the Jacobians of the transformation.
"""


import torch
import numpy as np

from abc import abstractmethod, ABC

from .._core import Module_
from ...lib.spline import RQSpline


class CouplingList_(Module_, ABC):
    """A base class for a list of invertible transformations using a mask-based
    approach to divide the input into two partitions that are coupled in a
    specific way that makes it easy to calculate the Jacobian of the
    transformation.

    The list contain several layers of couplings alternatively acting over each
    partition; each layer is specifiend by a NN in the `nets` list.

    Parameters
    ----------
    nets : a list of instance of torch.nn
         The output of each element of nets must have enought output channels
         for corresponding subclasses.

    mask : a tensor of 0s and 1s
         For partitioning the input.

    channels_axis : int, default 1
        The channel axis in the outputs of net0 and net1.

    label : str
        Can be used for unique labeling of NNs.
    """

    def __init__(self, nets, *, mask, channels_axis=1, label='cpl_', **kwargs):
        super().__init__(label=label)
        self.nets = torch.nn.ModuleList(nets)
        self.mask = mask
        self.channels_axis = channels_axis
        self.kwargs = kwargs

    def forward(self, x, log0=0):
        x = list(self.mask.split(x))  # x = [x_0, x_1]
        for k, net in enumerate(self.nets):
            parity = k % 2
            x[parity], log0 = self.atomic_forward(
                                                  x_active=x[parity],
                                                  x_frozen=x[1 - parity],
                                                  parity=parity,
                                                  net=net,
                                                  log0=log0
                                                  )
        return self.mask.cat(*x), log0

    def backward(self, x, log0=0):
        x = list(self.mask.split(x))  # x = [x_0, x_1]
        for k in list(range(len(self.nets)))[::-1]:
            parity = k % 2
            x[parity], log0 = self.atomic_backward(
                                                  x_active=x[parity],
                                                  x_frozen=x[1 - parity],
                                                  parity=parity,
                                                  net=self.nets[k],
                                                  log0=log0
                                                  )
        return self.mask.cat(*x), log0

    @abstractmethod
    def atomic_forward(self, *, x_active, x_frozen, parity, net, log0=0):
        pass

    @abstractmethod
    def atomic_backward(self, *, x_active, x_frozen, parity, net, log0=0):
        pass

    def preprocess_fz(self, x):  # fz: frozen
        return x.unsqueeze(self.channels_axis)

    def preprocess(self, x):
        return x.unsqueeze(self.channels_axis)

    def postprocess(self, x):
        return x.squeeze(self.channels_axis)

    def transfer(self, scale_factor=1, mask=None, **extra):
        return self.__class__(
                [net.transfer(scale_factor=scale_factor) for net in self.nets],
                mask=self.mask if mask is None else mask,
                label=self.label,
                channels_axis=self.channels_axis,
                **self.kwargs,
                )


class CntrCouplingList_(CouplingList_):
    """A "controlled" version of CouplingList_."""

    def forward(self, x_and_control, log0=0):
        x, control = x_and_control
        x, control = self.mask.split(x), self.mask.split(control)  # apply mask
        for k, net in enumerate(self.nets):
            parity = k % 2
            x_frozen = (control if k==0 else x)[1 - parity]
            x[parity], log0 = self.atomic_forward(
                                                  x_active=x[parity],
                                                  x_frozen=x_frozen,
                                                  parity=parity,
                                                  net=net,
                                                  log0=log0
                                                  )
        x_and_control = [self.mask.cat(x), control]
        return x_and_control, log0

    def backward(self, x_and_control, log0=0):
        x, control = x_and_control
        x, control = self.mask.split(x), self.mask.split(control)  # apply mask
        for k in list(range(len(self.nets)))[::-1]:
            parity = k % 2
            x_frozen = (control if k==0 else x)[1 - parity]
            x[parity], log0 = self.atomic_backward(
                                                  x_active=x[parity],
                                                  x_frozen=x_frozen,
                                                  parity=parity,
                                                  net=self.nets[k],
                                                  log0=log0
                                                  )
        x_and_control = [self.mask.cat(x), control]
        return x_and_control, log0


class ShiftList_(CouplingList_):
    """A coupling block with a shift transformation."""

    def atomic_forward(self, *, x_active, x_frozen, parity, net, log0=0):
        t = self.postprocess(net(self.preprocess_fz(x_frozen)))
        return self.mask.purify(x_active + t, channel=parity), log0

    def atomic_backward(self, *, x_active, x_frozen, parity, net, log0=0):
        t = self.postprocess(net(self.preprocess_fz(x_frozen)))
        return self.mask.purify(x_active - t, channel=parity), log0


class AffineList_(CouplingList_):
    """A coupling block with an affine transformation.

    It accepts a boolean kwargs called `zee2sym`;
    if True explicitely implements the Z_2 symmetry.
    """

    def atomic_forward(self, *, x_active, x_frozen, parity, net, log0=0):
        out = net(self.preprocess_fz(x_frozen))
        t, s = out.split((1, 1), dim=self.channels_axis)
        # purify: get rid of unwanted contributions to x_frozen
        t = self.mask.purify(self.postprocess(t), channel=parity)
        s = self.mask.purify(self.postprocess(s), channel=parity)
        if ('zee2sym' in self.kwargs.keys()) and self.kwargs['zee2sym']:
            s = torch.abs(s)
        return t + x_active * torch.exp(-s), log0 - self.sum_density(s)

    def atomic_backward(self, *, x_active, x_frozen, parity, net, log0=0):
        out = net(self.preprocess_fz(x_frozen))
        t, s = out.split((1, 1), dim=self.channels_axis)
        # purify: get rid of unwanted contributions to x_frozen
        t = self.mask.purify(self.postprocess(t), channel=parity)
        s = self.mask.purify(self.postprocess(s), channel=parity)
        if ('zee2sym' in self.kwargs.keys()) and self.kwargs['zee2sym']:
            s = torch.abs(s)
        return (x_active - t) * torch.exp(s), log0 + self.sum_density(s)


class RQSplineList_(CouplingList_):
    """A coupling block with a rational quadratic spline transformation."""

    def __init__(self, nets, *, mask,
            xlim=(0, 1), ylim=(0, 1), knots_x=None, knots_y=None, extrap={},
            channels_axis=1, label='spline_cpl_'
            ):
        """
        Tips on extrapolation:
        1.  for linear extrapolation on both sides set
            `extrap=dict(left='linear', right='linear')`
        2.  for linear extrapolation on right and anti-periodic boundary on left
            set `extrap={'left': 'anti', 'right': 'linear'}`.
        """

        super().__init__(nets,
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

    def atomic_forward(self, net, *, x_active, x_frozen, parity, log0=0):
        out = net(self.preprocess_fz(x_frozen))
        spline = self.make_spline(out)
        # below g is the gradient of spline @ x_active
        fx_active, g = spline(self.preprocess(x_active), grad=True)
        fx_active, g = self.postprocess(fx_active), self.postprocess(g)
        # the above two lines are equivalent to the following for default cases
        # fx_active, g = spline(x_active, grad=True, squeezed=True)
        fx_active = self.mask.purify(fx_active, channel=parity)
        g = self.mask.purify(g, channel=parity, zero2one=True)
        return fx_active, log0 + self.sum_density(torch.log(g))

    def atomic_backward(self, net, *, x_active, x_frozen, parity, log0=0):
        out = net(self.preprocess_fz(x_frozen))
        spline = self.make_spline(out)
        # below g is the gradient of spline @ x_active
        fx_active, g = spline.backward(self.preprocess(x_active), grad=True)
        fx_active, g = self.postprocess(fx_active), self.postprocess(g)
        # the above two lines are equivalent to the following for default cases
        # fx_active, g = spline.backward(x_active, grad=True, squeezed=True)
        fx_active = self.mask.purify(fx_active, channel=parity)
        g = self.mask.purify(g, channel=parity, zero2one=True)
        return fx_active, log0 + self.sum_density(torch.log(g))

    def _hack(self, x, parity=0):
        x_0, x_1 = self.mask.split(x)
        if parity == 0:
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
            return torch.zeros(pad_shape, device=w.device)

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
                channels_axis=self.channels_axis,
                xlim=self.xlim,
                ylim=self.ylim,
                knots_x=self.knots_x,
                knots_y=self.knots_y,
                extrap=self.extrap
                )


class MultiRQSplineList_(CouplingList_):
    """
    a coupling block with two rational quadratic spline transformations,
    transforming two different parameters.
    """

    def __init__(self, nets, *, mask,
            xlims=[(0, 1), (0, 1)], ylims=[(0, 1), (0, 1)],
            knots_x=[None, None], knots_y=[None, None],
            extraps=[{}, {}], channels_axis=1, label='mulspline_cpl_'
            ):
        """
        Tips on extrapolation:
        1.  for linear extrapolation on both sides set
            `extrap=dict(left='linear', right='linear')`
        2.  for linear extrapolation on right and anti-periodic boundary on left
            set `extrap={'left': 'anti', 'right': 'linear'}`.
        """

        super().__init__(nets,
                mask=mask, channels_axis=channels_axis, label=label
                )

        self.num_splines = len(xlims)
        self.xlims = xlims
        self.ylims = ylims
        self.xwidths = [xlim[1] - xlim[0] for xlim in xlims]
        self.ywidths = [ylim[1] - ylim[0] for ylim in ylims]
        self.knots_x = knots_x
        self.knots_y = knots_y
        self.extraps = extraps

        self.softmax = torch.nn.Softmax(dim=self.channels_axis)
        self.softplus = torch.nn.Softplus(beta=np.log(2))
        # we set the beta of Softplus to log(2) so that self.softplus(0)
        # returns 1. With this setting it would be easy to set the derivatives
        # to 1 (with zero inputs).

    def atomic_forward(self, *, x_active, x_frozen, parity, net, log0=0):
        out = net(self.preprocess_fz(x_frozen))
        spline = self.make_spline(out)
        # below g is the gradient of spline @ x_active
        fx_active, g = self.apply_spline(self.preprocess(x_active), spline)
        fx_active, g = self.postprocess(fx_active), self.postprocess(g)
        fx_active = self.mask.purify(fx_active, channel=parity)
        g = self.mask.purify(g, channel=parity, zero2one=True)
        return fx_active, log0 + self.sum_density(torch.log(g))

    def atomic_backward(self, *, x_active, x_frozen, parity, net, log0=0):
        out = net(self.preprocess_fz(x_frozen))
        spline = self.make_spline(out)
        # below g is the gradient of spline @ x_active
        fx_active, g = self.apply_spline(self.preprocess(x_active), spline, backward=True)
        fx_active, g = self.postprocess(fx_active), self.postprocess(g)
        fx_active = self.mask.purify(fx_active, channel=parity)
        g = self.mask.purify(g, channel=parity, zero2one=True)
        return fx_active, log0 + self.sum_density(torch.log(g))

    def preprocess(self, x):
        xs = torch.tensor_split(x, sections=self.num_splines, dim=self.channels_axis)
        return xs

    def postprocess(self, xs):
        # concatenate list of x_active channels into single tensor
        x = torch.cat(xs, dim=self.channels_axis)
        return x

    def make_spline(self, out):
        """splits the out in self.channel_axis into two equal parts and makes two splines,
        one for each independent parameter of the eigenvectors of the SU(3) matrix.
        """
        out_splits = torch.tensor_split(
                out, sections=self.num_splines, dim=self.channels_axis
                )

        axis = self.channels_axis
        def zeropad(w):
            pad_shape = list(w.shape)
            pad_shape[axis] = 1  # note that axis migh be e.g. -1
            return torch.zeros(pad_shape, device=w.device)

        cumsumsoftmax = lambda w: torch.cumsum(self.softmax(w), dim=axis)
        to_coord = lambda w: torch.cat((zeropad(w), cumsumsoftmax(w)), dim=axis)
        to_deriv = lambda d: self.softplus(d) if d is not None else None

        splines = []
        for i, out in enumerate(out_splits):
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
            knots_x, knots_y = self.knots_x[i], self.knots_y[i]
            xwidth, ywidth = self.xwidths[i], self.ywidths[i]
            xlim, ylim = self.xlims[i], self.ylims[i]
            extrap = self.extraps[i]

            n = out.shape[axis]  # n parameters to specify splines
            if knots_x is None and knots_y is None:
                m = (n + 2) // 3
                x_, y_, d_ = out.split((m-1, m-1, m), dim=axis)
                knots_x = to_coord(x_) * xwidth + xlim[0]
                knots_y = to_coord(y_) * ywidth + ylim[0]
                knots_d = to_deriv(d_)
            elif knots_x is not None and knots_y is None:
                m = (n + 2) // 2
                y_, d_ = out.split((m-1, m), dim=axis)
                knots_y = to_coord(y_) * ywidth + ylim[0]
                knots_d = to_deriv(d_)
            elif knots_x is None and knots_y is not None:
                m = (n + 2) // 2
                x_, d_ = out.split((m-1, m), dim=axis)
                knots_x = to_coord(x_) * xwidth + xlim[0]
                knots_d = to_deriv(d_)
            else:
                knots_d = to_deriv(out)

            kwargs = dict(knots_x=knots_x, knots_y=knots_y, knots_d=knots_d)
            kwargs.update(dict(knots_axis=axis, extrap=extrap))

            splines.append(RQSpline(**kwargs))

        return splines

    def apply_spline(self, x_actives, splines, backward=False):
        x_actives_out = []
        gs = []
        for i, x_active in enumerate(x_actives):
            transformation = splines[i].backward if backward else splines[i]
            x_active, g = transformation(x_active, grad=True)
            x_actives_out.append(x_active)
            gs.append(g)
        return x_actives_out, gs

    def transfer(self, scale_factor=1, mask=None, **extra):
        return self.__class__(
                self.net0.transfer(scale_factor=scale_factor),
                self.net1.transfer(scale_factor=scale_factor),
                mask=self.mask if mask is None else mask,
                label=self.label,
                channels_axis=self.channels_axis,
                xlim=self.xlims,
                ylim=self.ylims,
                knots_x=self.knots_x,
                knots_y=self.knots_y,
                extrap=self.extraps
                )


class CntrShiftList_(CntrCouplingList_, ShiftList_):
    pass


class CntrAffineList_(CntrCouplingList_, AffineList_):
    pass


class CntrRQSplineList_(CntrCouplingList_, RQSplineList_):
    pass


class CntrMultiRQSplineList_(CntrCouplingList_, MultiRQSplineList_):
    pass
