# Copyright (c) 2021-2023 Javad Komijani

"""This module is for introducing unitary priors."""


from .prior import Prior
from ..lib.stats import UnGroup, SUnGroup, U1Group


class UnPrior(Prior):
    """Generate random unitary matrices, i.e. random U(n)."""

    def __init__(self, *, n, shape=(1,), **kwargs):
        dist = UnGroup(n=n, shape=shape)
        super().__init__(dist, **kwargs)
        self.shape = shape

    def to(self, *args, **kwargs):
        """
        Moves the distibution parameters to a device, implying that the samples
        also will also be created on the same device.
        """
        dist = self.dist.normal_dist
        dist.loc = dist.loc.to(*args, **kwargs)
        dist.scale = dist.scale.to(*args, **kwargs)

    @property
    def parameters(self):
        """Returns all parameters needed to define the prior in a dict."""
        dist = self.dist.normal_dist
        return dict(loc=dist.loc, scale=dist.scale)


class SUnPrior(Prior):
    """Generate random special unitary matrices, i.e. random SU(n)."""

    def __init__(self, *, n, shape=(1,), **kwargs):
        dist = SUnGroup(n=n, shape=shape)
        super().__init__(dist, **kwargs)
        self.shape = shape

    def to(self, *args, **kwargs):
        """
        Moves the distibution parameters to a device, implying that the samples
        also will also be created on the same device.
        """
        dist = self.dist.normal_dist
        dist.loc = dist.loc.to(*args, **kwargs)
        dist.scale = dist.scale.to(*args, **kwargs)

    @property
    def parameters(self):
        """Returns all parameters needed to define the prior in a dict."""
        dist = self.dist.normal_dist
        return dict(loc=dist.loc, scale=dist.scale)


class U1Prior(Prior):
    """Generate random unitary matrices, i.e. random U(1).

    This is a faster implementation of random U(1) than `UnPrior(n=1)`.
    """

    def __init__(self, shape=(1,), **kwargs):
        dist = U1Group(shape=shape)
        super().__init__(dist, **kwargs)
        self.shape = shape

    def to(self, *args, **kwargs):
        """
        Moves the distibution parameters to a device, implying that the samples
        also will also be created on the same device.
        """
        dist = self.dist.uniform_dist
        dist.loc = dist.loc.to(*args, **kwargs)
        dist.scale = dist.scale.to(*args, **kwargs)

    @property
    def parameters(self):
        """Returns all parameters needed to define the prior in a dict."""
        dist = self.dist.uniform_dist
        return dict(loc=dist.loc, scale=dist.scale)
