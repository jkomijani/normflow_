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
        # moves the distribution parameters to a device, which implies that
        # samples will also be created on that device
        self.dist.normal_dist.loc = self.dist.normal_dist.loc.to(*args, **kwargs)
        self.dist.normal_dist.scale = self.dist.normal_dist.scale.to(*args, **kwargs)

    @property
    def parameters(self):
        return dict(loc=self.dist.normal_dist.loc.item(), scale=self.dist.normal_dist.scale.item())


class SUnPrior(Prior):
    """Generate random special unitary matrices, i.e. random SU(n)."""

    def __init__(self, *, n, shape=(1,), **kwargs):
        dist = SUnGroup(n=n, shape=shape)
        super().__init__(dist, **kwargs)
        self.shape = shape

    def to(self, *args, **kwargs):
        # moves the distribution parameters to a device, which implies that
        # samples will also be created on that device
        self.dist.normal_dist.loc = self.dist.normal_dist.loc.to(*args, **kwargs)
        self.dist.normal_dist.scale = self.dist.normal_dist.scale.to(*args, **kwargs)

    @property
    def parameters(self):
        return dict(loc=self.dist.normal_dist.loc.item(), scale=self.dist.normal_dist.scale.item())


class U1Prior(Prior):
    """Generate random unitary matrices, i.e. random U(1).

    This is a faster implementation of random U(1) than `UnPrior(n=1)`.
    """

    def __init__(self, shape=(1,), **kwargs):
        dist = U1Group(shape=shape)
        super().__init__(dist, **kwargs)
        self.shape = shape

    def to(self, *args, **kwargs):
        # moves the distribution parameters to a device, which implies that
        # samples will also be created on that device
        self.dist.uniform_dist.loc = self.dist.uniform_dist.loc.to(*args, **kwargs)
        self.dist.uniform_dist.scale = self.dist.uniform_dist.scale.to(*args, **kwargs)

    @property
    def parameters(self):
        return dict(loc=self.dist.uniform_dist.loc.item(), scale=self.dist.uniform_dist.scale.item())
