# Copyright (c) 2021-2022 Javad Komijani

"""
This module has classes to generate random unitary and special unitary matrices.
"""

import torch
import numpy as np

from math import log, lgamma, pi  # lgamma: log gamma


# =============================================================================
class UnGroup:
    """Generate random unitary matrices, i.e. random U(n).
    
    Simliar to `[scipy.stats.unitary_group]`_, we follow `[Mezzadri]`_ to
    generate randam unitary matrices.

    Parameters
    ----------
    n : int
        Specifies the dimension n of the U(n) matrices

    shape : tuple (optional)
        Specifing the shape of tensor of random unitary matrices.
        Each sample would be of a tensor of size (*shape, n, n),
        where the last two dimensions construct unitary matrices.

    .. _[scipy.stats.unitary_group]:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.unitary_group.html

    .. _[Mezzadri]:
        F. Mezzadri,
        "How to generate random matrices from the classical compact groups",
        :arXiv:`math-ph/0609050`.
    """

    def __init__(self, n, shape=(1,)):
        self.n = n
        self.shape = shape
        # shape_: shape of underlying torch tensor
        if n == 1:
            shape_ = shape
        else:
            shape_ = (*shape, n, n)

        if len(shape) == 1 and shape[0] == 1 and n > 1:
            shape_ = shape_[1:]
            self.reduced_shape = True
        else:
            self.reduced_shape = False

        loc = torch.zeros(shape_)  # i.e. mean
        scale = torch.ones(shape_)  # i.e. sigma
        self.normal_dist = torch.distributions.normal.Normal(loc, scale)
        self.log_group_vol = self.calc_log_group_volume()
        self.log_tot_vol = self.log_group_vol + np.log(np.product(shape))

    def calc_log_group_volume(self):
        """Return volume of U(n); we use eq (5.16) of `[Mezzadri]`_

        .. _[Mezzadri]:
            F. Mezzadri,
            "How to generate random matrices from the classical compact groups",
            :arXiv:`math-ph/0609050`.
        """
        n = self.n
        logc = log(n) + (n+1) * log(2) + (n**2 + n) * log(pi)
        return 0.5 * logc + sum([-lgamma(1+k) for k in range(1, n)])

    def log_prob(self, x):
        """Return log_prob of each matrix."""  # one value for each matrix
        shape = x.shape
        if self.n > 1:
            shape = shape[:-2]  # one log_prop for each matrix
        if self.reduced_shape:
            shape = (*shape, 1)  # restore the reduced dimension

        return torch.zeros(shape) - self.log_group_vol

    def sample(self, size=(1,)):  # this is the `sample` of dist (not prior)
        """Draw random samples."""

        normal = self.normal_dist.sample

        mat = normal(size) + 1j * normal(size)

        if self.n == 1:
            return mat / torch.abs(mat)
        else:
            q, r = torch.linalg.qr(mat, mode='complete')
            d = torch.diagonal(r, dim1=-2, dim2=-1).unsqueeze(-2)
            return q * (d/torch.abs(d))


# =============================================================================
class SUnGroup(UnGroup):
    """Generate random special unitary matrices, i.e. random SU(n).

    As a subgroup of unitary matrices, this class uses the `UnGroup` class
    to generate random SU(n) matrices.
    """

    def calc_log_group_volume(self):
        """Return volume of SU(n); we use eq (5.13) of `[Mezzadri]`_

        .. _[Mezzadri]:
            F. Mezzadri,
            "How to generate random matrices from the classical compact groups",
            :arXiv:`math-ph/0609050`.
        """
        n = self.n
        logc = log(n) + (n-1) * log(2) + (n**2 + n - 2) * log(pi)
        return 0.5 * logc + sum([-lgamma(1+k) for k in range(1, n)])

    def sample(self, size=(1,)):  # this is the `sample` of dist (not prior)
        """Draw random samples."""
        samples = super().sample(size)
        det = torch.linalg.det(samples).unsqueeze(-1).unsqueeze(-1)
        # Remarks:
        # 1. det \in U(1)
        # 2. det^(1/n) covers only 1/n-th of U(1) -> volume 1/n-th of U(1)
        # 3. determinant of det^(1/n) I_{n x n} covers U(1); stretching fac. n
        return samples / torch.pow(det, 1/self.n)


# =============================================================================
class U1Group:
    """Generate random unitary matrices, i.e. random U(1).

    This is an implementation of random U(1), which is faster than `UnGroup(1)`.

    For a random unitary variable, the probability distribution function
    is math:`p(z) = 1/(2 \pi i) 1/z` such that math:`p(z) dz` is always real
    and its integral over any circle about the origin amounts to unity.

    Parameters
    ----------
    shape : tuple (optional)
        Specifing the shape of multivariate unitary distribution.
    """

    def __init__(self, shape=(1,)):
        low = torch.zeros(shape)
        high = torch.ones(shape) * (2 * pi)
        self.shape = shape
        self.dist = torch.distributions.uniform.Uniform(low, high)
        self.log_group_vol = np.log(2 * pi)
        self.log_tot_vol = np.log(2 * pi) + np.log(np.product(shape))

    def sample(self, size=(1,)):  # this is the `sample` of dist (not prior)
        return torch.exp(1j * self.dist.sample(size))

    def log_prob(self, x):
        """This is the real part of math:`log(p(z))`, ie math:`log(|p(z)|)`."""
        return torch.zeros(x.shape) - self.log_group_vol


# =============================================================================
def test_spectrum(prior, n_samples=100, display=True, bins=30):
    """Test the distibution of eigenvalues.

    Since Haar measure is the analogue of a uniform distribution, each set of
    eigenvalues must have the same weight, therefore the normalized eigenvalue
    density is :math:`\rho(\theta) = 1 / (2 \pi)` `[Mezzadri]`_.
    The histogram generated here must agree with Figure 2.a of `[Mezzadri]`_.
     
    .. _[Mezzadri]:
        F. Mezzadri,
        "How to generate random matrices from the classical compact groups",
        :arXiv:`math-ph/0609050`.
    """
    x = prior.sample(n_samples)
    eig, _ = torch.linalg.eig(x)
    phase = torch.angle(eig)

    if display:
        import matplotlib.pyplot as plt
        import numpy as np

        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        axs[0].hist(phase.ravel().numpy(), bins=bins, density=True)
        axs[0].plot([-np.pi, np.pi], [1/2/np.pi]*2, ':k')
        axs[0].set_xlabel(r"$\theta$")
        axs[0].set_ylabel(r"$\rho(\theta)$")
        axs[0].set_xlim([-np.pi, np.pi])

        if len(phase.shape) > 1 and phase.shape[1] > 1:
            phase = torch.sort(phase, dim=-1)[0]
            diff_phase = phase[..., 1:] - phase[..., :-1]
            for ind in range(diff_phase.shape[-1]):
                diff = diff_phase[..., ind]
                axs[1].hist(diff.ravel().numpy(), bins=bins, density=True)
            axs[1].set_xlabel(r"$\theta$")
            axs[1].set_ylabel(r"$\rho(\Delta \theta)$")
            axs[1].set_xlim([0, 2 * np.pi])

    out = dict(eig=eig, prod=torch.prod(eig, -1), phase=phase)

    if display:
        return axs, out
    else:
        return out
