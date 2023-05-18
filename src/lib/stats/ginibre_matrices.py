# Copyright (c) 2021-2023 Javad Komijani

"""
This module has a class to generate complex matrices from the Ginibre ensemble.
"""

import torch


# =============================================================================
class GinibreComplexMatrixDist:
    """
    For generating matrices from the Ginibre ensemble, which is the ensemble of
    square, complex-values matrices with i.i.d. normal entries.

    For the definition see `[Mezzadri]`_, which takes advantage of the Ginibre
    random matrices to generate unitary matrices.

    Parameters
    ----------
    n : int
        Specifies the dimension n of the Ginibre matrices

    shape : tuple (optional)
        Specifing the shape of tensor of Ginibre matrices.
        Each sample would be of a tensor of size (*shape, n, n),
        where the last two dimensions construct GL(n, C) matrices.

    .. _[Mezzadri]:
        F. Mezzadri,
        "How to generate random matrices from the classical compact groups",
        :arXiv:`math-ph/0609050`.
    """

    def __init__(self, n, shape=(1,), sigma=1):
        self.n = n
        self.shape = shape
        shape_ = (*shape, n, n)  # the shape of underlying torch tensor
        loc = torch.zeros(shape_)  # i.e. mean
        scale = sigma * torch.ones(shape_)  # i.e. sigma
        self.normal_dist = torch.distributions.normal.Normal(loc, scale)

    def sample(self, size=(1,)):  # this is the `sample` of dist (not prior)
        """Draw random samples."""
        fnc = self.normal_dist.sample
        return fnc(size) + 1j * fnc(size)

    def log_prob(self, x):
        """Return log_prob for each matrix."""  # one value for each matrix
        fnc = self.normal_dist.log_prob
        return torch.sum(fnc(x.real) + fnc(x.imag), dim=(-2, -1))
