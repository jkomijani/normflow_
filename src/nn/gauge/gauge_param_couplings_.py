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
    """A Coupling_ with a transformations as a Pade approximant of order 1/1

    .. math::

        f(x; t) = x / (x + e^t * (1 - x))

    Note that :math:`f(.; 1/t)` is the inverse of :math:`f(.; t)`.

    This transformation is useful when the input and output variables vary
    between zero and one.

    This transformation is equivalent to

    .. math::

        f(x; t) = \expit(\logit(x) - t)
    """
    def atomic_forward(self, *, x_active, x_frozen, parity, net, log0=0):
        t = net(x_frozen)
        t = self.mask.purify(t, channel=parity)
        denom = x_active + torch.exp(t) * (1 - x_active)
        y = x_active / denom
        logJ = self.sum_density(t - 2 * torch.log(denom))
        return y, log0 + logJ

    def atomic_backward(self, *, x_active, x_frozen, parity, net, log0=0):
        t = net(x_frozen)
        t = self.mask.purify(t, channel=parity)
        denom = x_active + torch.exp(-t) * (1 - x_active)
        y = x_active / denom
        logJ = self.sum_density(-t - 2 * torch.log(denom))
        return y, log0 + logJ


class Pade22Coupling_(Coupling_):
    pass


# =============================================================================
class Pade11DualCoupling_(Module_):
    """A Coupling_ with a transformations as a Pade approximant of order 1/1

    .. math::

        f(x; t) = x / (x + e^t * (1 - x))

    Note that :math:`f(.; 1/t)` is the inverse of :math:`f(.; t)`.

    This transformation is useful when the input and output variables vary
    between zero and one.

    This transformation is equivalent to

    .. math::

        f(x; t) = \expit(\logit(x) - t)
    """
    def __init__(self, net, *, mask, label='coupling_'):
        super().__init__(label=label)
        self.net = net
        self.mask = mask

    def forward(self, x, s, log0=0):
        x_active, x_passive = self.mask.split(x)
        s_frozen, s_passive = self.mask.split(s)

        t = self.net(s_frozen)
        t = self.mask.purify(t, channel=0)

        denom = x_active + torch.exp(t) * (1 - x_active)
        x_active = x_active / denom
        logJ = self.sum_density(t - 2 * torch.log(denom))

        return self.mask.cat(x_active, x_passive), log0 + logJ

    def backward(self, x, s, log0=0):
        x_active, x_passive = self.mask.split(x)
        s_frozen, s_passive = self.mask.split(s)

        t = self.net(s_frozen)
        t = self.mask.purify(t, channel=0)

        denom = x_active + torch.exp(-t) * (1 - x_active)
        x_active = x_active / denom
        logJ = self.sum_density(-t - 2 * torch.log(denom))

        return self.mask.cat(x_active, x_passive), log0 + logJ


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
