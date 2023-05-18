# Copyright (c) 2021-2023 Javad Komijani

"""This module is for introducing priors for random matrices."""


from .prior import Prior
from ..lib.stats import GinibreComplexMatrixDist


class GinibrePrior(Prior):
    """Generate random matrices from the Ginibre ensemble."""

    def __init__(self, *, n, shape=(1,), sigma=1, **kwargs):
        dist = GinibreComplexMatrixDist(n, shape=shape, sigma=sigma)
        super().__init__(dist, **kwargs)
        self.shape = shape

    def to(self, *args, **kwargs):
        # moves the distribution parameters to a device, which implies that
        # samples will also be created on that device
        dist = self.dist.normal_dist
        dist.loc = dist.loc.to(*args, **kwargs)
        dist.scale = dist.scale.to(*args, **kwargs)

    @property
    def parameters(self):
        dist = self.dist.normal_dist
        return dict(loc=dist.loc, scale=dist.scale)
