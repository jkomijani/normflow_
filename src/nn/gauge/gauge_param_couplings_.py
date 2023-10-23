# Copyright (c) 2021-2023 Javad Komijani

"""
This module contains new neural networks that are subclasses of `Module_`
and couple sites to each other.
"""


import torch
import numpy as np

from .._core import Module_
from ..scalar.modules_ import Logit_, Expit_
from ..scalar.couplings_ import AffineCoupling_, Coupling_

pi = np.pi


# =============================================================================
class Pade11Coupling_(Coupling_):
    r"""A transformations as a Pade approximant of order 1/1

    .. math::

        f(x; t) = x / (x + e^t * (1 - x))

    that maps :math:`[0, 1] \to [0, 1]` and is useful for input and output
    variables that vary between zero and one.

    This transformation is equivalent to math:`f(x; t) = \expit(\logit(x) - t)`
    and its inverse is :math:`f(.; -t)`.
    """
    def atomic_forward(self, *, x_active, x_frozen, parity, net, log0=0):
        t = net(x_frozen)
        t = self.mask.purify(t, channel=parity)
        denom = x_active + torch.exp(t) * (1 - x_active)
        x_active = x_active / denom
        logJ = self.sum_density(t - 2 * torch.log(denom))
        return x_active, log0 + logJ

    def atomic_backward(self, *, x_active, x_frozen, parity, net, log0=0):
        t = net(x_frozen)
        t = self.mask.purify(t, channel=parity)
        denom = x_active + torch.exp(-t) * (1 - x_active)
        x_active = x_active / denom
        logJ = self.sum_density(-t - 2 * torch.log(denom))
        return x_active, log0 + logJ


# =============================================================================
class Pade22Coupling_(Coupling_):
    r"""A transformations as an invertible Pade approximant of order 2/2

    .. math::

        f(x; t) = (x^2 + a x (1 - x)) / (1 + b x (1 - x))

    that maps :math:`[0, 1] \to [0, 1]` and is useful for input and output
    variables that vary between zero and one.
    """
    def atomic_forward(self, *, x_active, x_frozen, parity, net, log0=0):
        t = net(x_frozen)
        t = self.mask.purify(t, channel=parity)
        d0, d1 = torch.exp(t).chunk(2, dim=self.channels_axis)

        def pade22_(x):
            denom = (1 + (d1 + d0 - 2) * x * (1 - x))
            g_0 = x * (x + d0 * (1 - x)) / denom
            g_1 = (d0 + 2 * (1 - d0) * x + (d1 + d0 - 2) * x**2) / denom**2
            return g_0, self.sum_density(torch.log(g_1))

        x_active, logJ = pade22_(x_active)

        return x_active, log0 + logJ

    def atomic_backward(self, *, x_active, x_frozen, parity, net, log0=0):
        t = net(x_frozen)
        t = self.mask.purify(t, channel=parity)
        d0, d1 = torch.exp(t).chunk(2, dim=self.channels_axis)

        def invert(y):
            # returns the positive solution of $a x^2 + b x + c = 0$
            c = y
            b = (d1 + d0 - 2) * y - d0
            a = -1 - b
            delta = torch.sqrt(b**2 - 4 * c * a)
            x = (-b - delta) / (2 * a)
            x[a == 0] = (-c / b)[a == 0]
            return x

        def invpade22_(y):
            x = invert(y)
            denom = (1 + (d1 + d0 - 2) * x * (1 - x))
            g_1 = (d0 + 2 * (1 - d0) * x + (d1 + d0 - 2) * x**2) / denom**2
            return x, - self.sum_density(torch.log(g_1))

        x_active, logJ = invpade22_(x_active)

        return x_active, log0 + logJ


# =============================================================================
class Pade11DualCoupling_(Module_):
    r"""A transformations as a Pade approximant of order 1/1

    .. math::

        f(x; t) = x / (x + e^t * (1 - x))

    that maps :math:`[0, 1] \to [0, 1]` and is useful for input and output
    variables that vary between zero and one.

    This transformation is equivalent to math:`f(x; t) = \expit(\logit(x) - t)`
    and its inverse is :math:`f(.; -t)`.
    """
    def __init__(self, net, *, mask, label='p11_dualcoupl_'):
        super().__init__(label=label)
        self.net = net
        self.mask = mask

    def forward(self, x, s, log0=0):
        x_active, x_invisible = self.mask.split(x)
        s_frozen, s_invisible = self.mask.split(s)

        t = self.net(s_frozen)
        t = self.mask.purify(t, channel=0)

        denom = x_active + torch.exp(t) * (1 - x_active)
        x_active = x_active / denom
        logJ = self.sum_density(t - 2 * torch.log(denom))

        return self.mask.cat(x_active, x_invisible), log0 + logJ

    def backward(self, x, s, log0=0):
        x_active, x_invisible = self.mask.split(x)
        s_frozen, s_invisible = self.mask.split(s)

        t = self.net(s_frozen)
        t = self.mask.purify(t, channel=0)

        denom = x_active + torch.exp(-t) * (1 - x_active)
        x_active = x_active / denom
        logJ = self.sum_density(-t - 2 * torch.log(denom))

        return self.mask.cat(x_active, x_invisible), log0 + logJ


# =============================================================================
class Pade22DualCoupling_(Module_):
    r"""A transformations as an invertible Pade approximant of order 2/2

    .. math::

        f(x; t) = (x^2 + a x (1 - x)) / (1 + b x (1 - x))

    that maps :math:`[0, 1] \to [0, 1]` and is useful for input and output
    variables that vary between zero and one.
    """
    def __init__(self, net, *, mask, channels_axis=1, label='p22_dualcoupl_'):
        super().__init__(label=label)
        self.net = net
        self.mask = mask
        self.channels_axis = channels_axis

    def forward(self, x, s, log0=0):
        x_active, x_invisible = self.mask.split(x)
        s_frozen, s_invisible = self.mask.split(s)

        t = self.net(s_frozen)
        t = self.mask.purify(t, channel=0)
        d0, d1 = torch.exp(t).chunk(2, dim=self.channels_axis)

        def pade22_(x):
            denom = (1 + (d1 + d0 - 2) * x * (1 - x))
            g_0 = x * (x + d0 * (1 - x)) / denom
            g_1 = (d0 + 2 * (1 - d0) * x + (d1 + d0 - 2) * x**2) / denom**2
            return g_0, self.sum_density(torch.log(g_1))

        x_active, logJ = pade22_(x_active)

        return self.mask.cat(x_active, x_invisible), log0 + logJ

    def backward(self, x, s, log0=0):
        x_active, x_invisible = self.mask.split(x)
        s_frozen, s_invisible = self.mask.split(s)

        t = self.net(s_frozen)
        t = self.mask.purify(t, channel=0)
        d0, d1 = torch.exp(t).chunk(2, dim=self.channels_axis)

        def invert(y):
            # returns the positive solution of $a x^2 + b x + c = 0$
            c = y
            b = (d1 + d0 - 2) * y - d0
            a = -1 - b
            delta = torch.sqrt(b**2 - 4 * c * a)
            x = (-b - delta) / (2 * a)
            x[a == 0] = (-c / b)[a == 0]
            return x

        def invpade22_(y):
            x = invert(y)
            denom = (1 + (d1 + d0 - 2) * x * (1 - x))
            g_1 = (d0 + 2 * (1 - d0) * x + (d1 + d0 - 2) * x**2) / denom**2
            return x, - self.sum_density(torch.log(g_1))

        x_active, logJ = invpade22_(x_active)

        return self.mask.cat(x_active, x_invisible), log0 + logJ


# =============================================================================
class SUnParamAffineCoupling_(AffineCoupling_):

    logit_ = Logit_()
    expit_ = Expit_()

    def forward(self, x, log0=0):
        return self.expit_(*super().forward(*self.logit_(x, log0=log0)))

    def backward(self, x, log0=0):
        return self.expit_(*super().backward(*self.logit_(x, log0=log0)))

    def preprocess_fz(self, x):
        return x

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return x
