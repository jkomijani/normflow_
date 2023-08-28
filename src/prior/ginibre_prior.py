# Copyright (c) 2021-2023 Javad Komijani

"""This module is for introducing priors for random matrices."""


from .prior import Prior
from ..lib.stats import GinibreCMatrixDist


class GinibrePrior(Prior):
    """Generate random matrices from the Ginibre ensemble."""

    def __init__(self, *, n, shape=(1,), sigma=1, **kwargs):
        dist = GinibreCMatrixDist(n=n, shape=shape, sigma=sigma)
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
