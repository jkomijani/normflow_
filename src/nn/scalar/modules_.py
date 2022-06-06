# Copyright (c) 2021-2022 Javad Komijani

"""This module contains new neural networks that
1. are children of Module_,
2. do NOT couple sites to each other.

Note that as in Module_ itself, the trailing underscore implies that
the associated forward and backward methods handle the Jacobians of
the transformation.
"""


import torch
import copy
import numpy as np

from .modules import SplineNet
from .._core import Module_, ModuleList_


class Identity_(Module_):

    def __init__(self, label='identity_'):
        super().__init__(label=label)

    def forward(self, x, log0=0, **extra):
        return x, log0

    def backward(self, x, log0=0, **extra):
        return x, log0


class ScaleNet_(Module_):
    """Scales the input by a constant factor available as self.logw"""

    def __init__(self, label='scale_'):
        super().__init__(label=label)
        self.logw = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, log0=0):
        return x * torch.exp(self.logw), log0 + self.log_jacobian(x.shape)

    def backward(self, x, log0=0):
        return x / torch.exp(self.logw), log0 - self.log_jacobian(x.shape)

    def log_jacobian(self, x_shape):
        if Module_._propagate_density:
            return self.logw * torch.ones(x_shape)
        else:
            return (self.logw * np.product(x_shape[1:])) * torch.ones(x_shape[0])


class Tanh_(Module_):

    def forward(self, x, log0=0):
        logJ = -2 * self.sum_density(torch.log(torch.cosh(x)))
        return torch.tanh(x), log0 + logJ

    def backward(self, x, log0=0):
        return ArcTanh_().forward(x, log0)


class ArcTanh_(Module_):

    def forward(self, x, log0=0):
        y = torch.atanh(x)
        logJ = 2 * self.sum_density(torch.log(torch.cosh(y)))
        return y, log0 + logJ

    def backward(self, x, log0=0):
        return Tanh_().forward(x, log0)


class Expit_(Module_):
    """This can be also called Sigmoid_"""

    def forward(self, x, log0=0):
        y = 1/(1 + torch.exp(-x))
        sum_dim = tuple(range(1, len(x.size())))
        logJ = self.sum_density(-x + 2 * torch.log(y))
        return y, log0 + logJ

    def backward(self, x, log0=0):
        return Logit_().forward(x, log0)


class Logit_(Module_):
    """This is inverse of Sigmoid_"""

    def forward(self, x, log0=0):
        y = torch.log(x/(1 - x))
        logJ = - self.sum_density(torch.log(x * (1 - x)))
        return y, log0 + logJ

    def backward(self, x, log0=0):
        return Expit_().forward(x, log0)


class SplineNet_(SplineNet, Module_):
    """Identical to SplineNet, except for taking care of log_jacobian.

    This can be used as a probability distribution convertor for variables with
    nonzero probability in [0, 1].
    """

    def forward(self, x, log0=0):
        spline = self.make_spline()
        fx, g = spline(x.ravel(), grad=True)  # g is gradient of the spline @ x
        fx, g = fx.reshape(x.shape), g.reshape(x.shape)
        logJ = self.sum_density(torch.log(g))
        return fx, log0 + logJ

    def backward(self, x, log0=0):
        spline = self.make_spline()
        fx, g = spline.backward(x.ravel(), grad=True)  # g is gradient @ x
        fx, g = fx.reshape(x.shape), g.reshape(x.shape)
        logJ = self.sum_density(torch.log(g))
        return fx, log0 + logJ


UnityDistConvertor_ = SplineNet_  # for PDF convertor, with variable in [0, 1].


class PhaseDistConvertor_(SplineNet_):
    """A phase probability distribution convertor, suitable for variables
    with nonzero probability in [-pi, pi].
    """

    def __init__(self, knots_len, symmetric=False, label='phase-dc_', **kwargs):

        pi = np.pi

        if symmetric:
            extra = dict(xlim=(0, pi), ylim=(0, pi), extrap={'left':'anti'})
        else:
            extra = dict(xlim=(-pi, pi), ylim=(-pi, pi))

        super().__init__(knots_len, label=label, **kwargs, **extra)


class DistConvertor_(ModuleList_):
    """A probability distribution convertor, suitable for variables potentially
    spread to plus/minus infinities.

    Steps: pass through Expit_, SplineNet_, and Logit_
    """

    def __init__(self, knots_len, symmetric=False, label='dc_',
            sgnbias=False, initial_scale=False, final_scale=False,
            **kwargs
            ):

        if symmetric:
            extra = dict(xlim=(0.5, 1), ylim=(0.5, 1), extrap={'left':'anti'})
        else:
            extra = dict(xlim=(0, 1), ylim=(0, 1))

        if knots_len > 1:
            spline_ = SplineNet_(knots_len, label='spline_', **kwargs, **extra)
            nets_ = [Expit_(label='expit_'), spline_, Logit_(label='logit_')]
        else:
            nets_ = []

        if initial_scale:
            nets_ = [ScaleNet_(label='scale_')] + nets_
        elif final_scale:
            nets_ = nets_ + [ScaleNet_(label='scale_')]

        if sgnbias:  # SgnBiasNet_() **must** come first if exits
            nets_ = [SgnBiasNet_()] + nets_

        super().__init__(nets_)
        self.label = label

    def cdf_mapper(self, x):
        """Useful for mapping the CDF of inputs to the CDF of outputs."""
        # The input `x` is expected to be in range 0 to 1.
        self.get_spline_.forward(x)

    def transfer(self, **kwargs):
        return copy.deepcopy(self)

    @property
    def get_spline_(self):
        for net_ in self:
            if net_.label == 'spline_':
                return net_

    @property
    def get_scale_(self):
        for net_ in self:
            if net_.label == 'scale_':
                return net_

    @property
    def get_sgnbias_(self):
        for net_ in self:
            if net_.label == 'sgnbias_':
                return net_


class SgnBiasNet_(Module_):
    """This module should be used only and only in the first layer, where the
    input does not depend on the parameters of the net. Otherwise, because it
    is not continuous, the derivatives will be messed up.
    """

    def __init__(self, size=[1], label='sgnbias_'):
        super().__init__(label=label)
        self.w = torch.nn.Parameter(torch.rand(*size)/10)

    def forward(self, x, log0=0):
        return x + torch.sgn(x) * self.w**2, log0

    def backward(self, x, log0=0):
        return x - torch.sgn(x) * self.w**2, log0
