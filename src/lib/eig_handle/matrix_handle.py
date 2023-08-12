# Copyright (c) 2021-2022 Javad Komijani

"""This module has utilities to deal with eigenvalues of matrices.

For SU(n) matrices, the main class is `SUnMatrixParametrizer` that has several
methods, including methods to map matrices to phases (of eigenvalues) and
parametrize the phases. The reverse of these maps are also defined.
"""


import torch
import numpy as np

from .ordering import ZeroSumOrder, ModalOrder
from ..linalg import eigsu  # eig for speical unitray matrices


eigu = eigsu  # For now we deal with SU(n), but later we may need to change it

mul = torch.matmul
pi = np.pi


# =============================================================================
class UnMatrixParametrizer:

    def __init__(self):
        self.modal_matrix = None
        self.phase = None
        self.order = None  # an object to sort the eigen-phases

    def free_memory(self):
        self.modal_matrix = None
        self.phase = None
        self.order = None  # an object to sort the eigen-phases

    def matrix2phase_(self, matrix):
        """Return angle of eigenvalues of input and logJ of transformation.

        `logJ` is the Jacobian of partitioning an integration over SU(n)
        matrices to integrals over corresponding spectral and modal matrices.
        The inverse of Jacobian is equal to the volume of conjugacy class.
        """
        eig, self.modal_matrix = eigu(matrix)  # torch.linalg.eig(matrix)
        self.phase = torch.angle(eig)  # in (-pi, pi]
        # we save phase because it can be useful when self.param2matrix_ is
        # called with the `reduce_` option set to True

        # Note: when |eig| = 1, logJ of eig to phase conversion is zero;
        # thus, we only need to take care of Jacobian of spectral decomposition:
        # *inverse* of Jacobian equals the volume of conjugacy class
        logJ = -sum_density(self.calc_log_conjugacy_vol(eig))  # up to a const.

        return self.phase, logJ

    def phase2matrix_(self, phase, reduce_=False):
        """Inverse of `self.matrix2phase_`.

        Return the matrix corresponding to `phase` and logJ of transformation.

        For the sake of frugal computing, the `reduce_` option is introduced
        such that if True, this method returns `M * M_old^\dagger`,
        where `M_old` is the matrix constructed with self.sorted_phase.
        """
        eig = exp1j(phase)
        modal = self.modal_matrix
        eig_prime = eig if not reduce_ else eig * exp1j(-self.phase)
        matrix = mul(modal, eig_prime.unsqueeze(-1) * modal.adjoint())

        # Note: when |eig| = 1, logJ of eig to phase conversion is zero;
        # thus, we only need to take care of Jacobian of spectral decomposition:
        # Jacobian equals the volume of conjugacy class
        logJ = sum_density(self.calc_log_conjugacy_vol(eig))  # up to a const.

        return matrix, logJ

    def phase2param_(self, *args, **kwargs):
        pass

    def param2phase_(self, *args, **kwargs):
        pass

    def matrix2param_(self, matrix):
        """Like matrix2phase_ except that returns a parametrization of phases."""
        phase, logJ_m2f = self.matrix2phase_(matrix)  # phase in (-pi, pi]
        param, logJ_f2p = self.phase2param_(phase)
        return param, logJ_m2f + logJ_f2p

    def param2matrix_(self, param, reduce_=False):
        """Like phase2matrix_ except that the input is a parametrization of
        phases.

        For the sake of frugal computing, the `reduce_` option is introduced
        such that if True, this method returns `M * M_old^\dagger`,
        where `M_old` is the matrix constructed with self.sorted_phase.
        """
        phase, logJ_p2f = self.param2phase_(param)
        matrix, logJ_p2m = self.phase2matrix_(phase, reduce_=reduce_)
        return matrix, logJ_p2f + logJ_p2m

    @staticmethod
    def calc_log_conjugacy_vol(eig):
        """Return log of conjugacy volume up to an additive constant."""
        sumlogabs2 = lambda x: 2 * torch.sum(torch.log(torch.abs(x)), dim=-1)
        log_vol = torch.zeros(eig.shape[:-1], device=eig.device)
        for k in range(eig.shape[-1] - 1):
            log_vol += sumlogabs2(eig[..., k:k+1] - eig[..., k+1:])
        return log_vol.unsqueeze(-1)  # unsqueeze to keep dimensions the same

    @staticmethod
    def calc_conjugacy_vol(eig):
        """Return conjugacy volume up to a multiplacative constant."""
        prodabs2 = lambda x: torch.prod(torch.abs(x)**2, dim=-1)
        vol = torch.ones(eig.shape[:-1])
        for k in range(eig.shape[-1] - 1):
            vol *= prodabs2(eig[..., k:k+1] - eig[..., k+1:])
        return vol.unsqueeze(-1)  # unsqueeze to keep dimensions the same


# =============================================================================
class SUnMatrixParametrizer(UnMatrixParametrizer):

    def phase2param_(self, phase):
        self.order = ModalOrder(self.modal_matrix)  # see order.sorted_ind
        sorted_phase = self.order.sort(phase)
        return self.sortedphase2param_(sorted_phase)

    def param2phase_(self, param):
        phase, logJ = self.param2sortedphase_(param)
        phase = self.order.revert(phase)  # revert the "sort" operation
        return phase, logJ

    @staticmethod
    def sortedphase2param_(sorted_phase):
        n = sorted_phase.shape[-1]
        return sorted_phase.split((1, n-1), dim=-1)[1], 0  # logJ = 0

    @staticmethod
    def param2sortedphase_(param):
        phase0 = -torch.sum(param, dim=-1).unsqueeze(-1)
        return torch.cat((phase0, param), dim=-1), 0  # logJ = 0


# =============================================================================
class SU2MatrixParametrizer(UnMatrixParametrizer):
    """Special case of SUnMatrixParametrizer with simpler methods."""

    def phase2param_(self, phase):
        self.order = ZeroSumOrder(phase)  # see order.(sorted_val & sorted_ind)
        return self.sortedphase2param_(self.order.sorted_val)

    def param2phase_(self, param, reduce_=False):
        phase, logJ = self.param2sortedphase_(param)
        phase = self.order.revert(phase)  # revert the "sort" operation
        return phase, logJ

    @staticmethod
    def sortedphase2param_(sorted_phase):
        """param changes between 0 and 1."""
        logJ = 0  # logJ = -np.log(pi) but suppress the additive constant
        return sorted_phase[..., 1:] / pi, logJ

    @staticmethod
    def param2sortedphase_(param):
        logJ = 0  # logJ = np.log(pi) but suppress the additive constant
        return torch.cat((-param * pi, param * pi), dim=-1), logJ


# =============================================================================
class SU3MatrixParametrizer(UnMatrixParametrizer):
    """Special case of SUnMatrixParametrizer with simpler methods."""

    def phase2param_(self, phase):
        self.order = ZeroSumOrder(phase)  # see order.(sorted_val & sorted_ind)
        return self.sortedphase2param_(self.order.sorted_val)

    def param2phase_(self, param, reduce_=False):
        phase, logJ = self.param2sortedphase_(param)
        phase = self.order.revert(phase)  # revert the "sort" operation
        return phase, logJ

    @staticmethod
    def sortedphase2param_(sorted_phase):
        r"""Return :math:`(w, r)` as defined below

        .. math::
            w = \theta \cos (\phi) / \pi \in [0, 1] \\
            r = \tan (\phi) \sqrt{3} \pi \in [-\pi, \pi]

        To convert w to \theta one can simply multiply it with
        :math:`\sqrt{1 + r^2 / (3 * \pi^2)}`.
        """
        x, y, z = sorted_phase.split((1, 1, 1), dim=-1)  # x <= y <= z
        w = (z - x) / (2 * pi)  # w \in [0, 1]
        r = (y / w) * (3/2)  # r \in [-pi, pi]
        r[w == 0] = 0
        # w *= torch.sqrt(1 + r**2 / (3 * pi**2)) * (3**0.5 / 2)
        c = np.log(2/3 * pi)
        return torch.cat((w, r), dim=-1), -sum_density(torch.log(w) + c)

    @staticmethod
    def param2sortedphase_(param):
        """Inverse of sortedphase2param_()"""
        w, r = param.split(1, dim=-1)
        y = w * r / (3/2)
        z = w * pi  - y/2
        x = -w * pi - y/2
        c = np.log(2/3 * pi)
        return torch.cat((x, y, z), dim=-1), sum_density(torch.log(w) + c)


# =============================================================================
class U1Parametrizer:
    """Properties and methods are chosen to be consistent with SU(n)."""

    def matrix2param_(self, u1, **kwargs):
        """Return angle of eigenvalues and logJ of transformation."""
        phase = torch.angle(u1)  # in (-pi, pi]
        self.phase = phase.unsqueeze(-1)  # to be consistent with SU(n)
        return self.phase, 0  # logJ = 0

    def param2matrix_(self, phase, reduce_=False):
        if reduce_:
            phase -= self.phase
        return exp1j(phase).squeeze(-1), 0


# =============================================================================
def sum_density(x):
    return torch.sum(x, dim=list(range(1, x.dim())))


def exp1j(phase):
    """Return `e^{i * phase}`."""
    return torch.exp(1J * phase)
