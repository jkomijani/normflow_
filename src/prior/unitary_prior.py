# Copyright (c) 2021-2022 Javad Komijani

"""This module is for introducing unitary priors."""


from .prior import Prior
from ..lib.stats.unitary_group import UnGroup, SUnGroup, U1Group


class UnPrior(Prior):
    """Generate random unitary matrices, i.e. random U(n)."""

    def __init__(self, *, n, shape=(1,), **kwargs):
        dist = UnGroup(n, shape)
        super().__init__(dist, **kwargs)
        self.shape = shape


class SUnPrior(Prior):
    """Generate random special unitary matrices, i.e. random SU(n)."""

    def __init__(self, *, n, shape=(1,), **kwargs):
        dist = SUnGroup(n, shape)
        super().__init__(dist, **kwargs)
        self.shape = shape


class U1Prior(Prior):
    """Generate random unitary matrices, i.e. random U(1).

    This is a faster implementation of random U(1) than `UnPrior(n=1)`.
    """

    def __init__(self, shape=(1,), **kwargs):
        dist = U1Group(shape)
        super().__init__(dist, **kwargs)
        self.shape = shape
