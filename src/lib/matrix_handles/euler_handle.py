# Copyright (c) 2023 Javad Komijani

"""This module has utilities to deal with euler decompostions of matrices."""


import torch
import numpy as np

from .matrix_handle import SU2MatrixParametrizer as SU2MatrixEigParametrizer
from ..linalg import su2_to_euler_angles, euler_angles_to_su2


mul = torch.matmul
pi = np.pi


# =============================================================================
class SU2MatrixEulerParametrizer:
    r"""For Euler decomposition of SU(2) matrices.

    We use `su2_to_euler_angles` for decomposition, which can be called with
    `alt_param` option set to False (default) or True:

        >>> [phi, theta, psi] = su2_to_euler_angles(matrix)
        >>> [abs00, s, d] = su2_to_euler_angles(matrix, alt_param=True)

    where
        >>> s = (phi + psi) / (2 np.pi) + 0.5  # sum
        >>> d = (phi - psi) / (2 np.pi) + 0.5  # diff
        >>> abs00 = cos(theta / 2)

    where all vary between zero and unity.

    Inside this class we always set `alt_param = True`.
    """

    def __init__(self, nonfixed_angles_list=[0, 1]):
        self.nonfixed_angles_list = nonfixed_angles_list
        self.euler_angles = [None, None, None]

    @classmethod
    def matrix2angles_(cls, matrix):
        """Calculate, save (for later use), and return the euler angles of
        the input matrix along with the Jacabian of the transformation.
        """
        euler_angles = su2_to_euler_angles(matrix, alt_param=True)
        logJ = sum_density(
                cls.calc_log_jacobian(euler_angles, alt_param=True)
                )  # up to an additive constant
        return euler_angles, logJ

    @classmethod
    def angles2matrix_(cls, euler_angles):
        """Reverse of matrix2angles_."""
        matrix = euler_angles_to_su2(*euler_angles, alt_param=True)
        logJ = -sum_density(
                cls.calc_log_jacobian(euler_angles, alt_param=True)
                )  # up to an additive constant
        return matrix, logJ

    def matrix2param_(self, matrix):
        """Return the euler angles normalized so that all are in [0, 1].
        
        The scaling of the euler angles is not incorporated to logJ because it
        is going to be undone when param2matrix_ is called.
        """
        self.euler_angles, logJ = self.matrix2angles_(matrix)
        euler_param = [self.euler_angles[k] for k in self.nonfixed_angles_list]
        return torch.stack(euler_param, dim=-1), logJ

    def param2matrix_(self, param, reduce_=False):
        """Reverse of matrix2param_.

        The `reduce_` option is introduced such that if True, this method
        returns `M * M_old^\dagger`.
        """
        euler_angles = [angle for angle in self.euler_angles]
        # now update the nonfixed ones from the input param
        for l, k in enumerate(self.nonfixed_angles_list):
            euler_angles[k] = param[..., l]

        new_matrix, logJ = self.angles2matrix_(euler_angles)

        if reduce_:
            old_matrix = euler_angles_to_su2(*self.euler_angles, alt_param=True)
            return new_matrix @ old_matrix.adjoint(), logJ
        else:
            return new_matrix, logJ

    @staticmethod
    def calc_log_jacobian(euler_angles, alt_param=True):
        """Return log of jacobian of transformation to euler angles up to an
        additive constant.

        See eq (4.8) in arXiv:math-ph/0210033.
        """
        # *inverse* of Jacobian equals the volume of conjugacy class
        if alt_param:
            abs00 = euler_angles[0]
            return -torch.log(abs00)
        else:
            theta = euler_angles[1]
            return -torch.log(torch.sin(theta))

    @staticmethod
    def calc_jacobian(euler_angles, alt_param=True):
        """Return jacobian of transformation to euler angles up to a
        multiplicative constant.

        See eq (4.8) in arXiv:math-ph/0210033.
        """
        if alt_param:
            abs00 = euler_angles[0]
            return 1 / abs00
        else:
            theta = euler_angles[1]
            return 1 / torch.sin(theta)


# =============================================================================
class SU2MatrixParametrizer:

    def __init__(self,
            eig_parametrizer=SU2MatrixEigParametrizer(),
            eul_parametrizer=SU2MatrixEulerParametrizer()
            ):
        self.eig_parametrizer = eig_parametrizer
        self.eul_parametrizer = eul_parametrizer

    def matrix2param_(self, matrix):

        eig_param, eig_logJ = self.eig_parametrizer.matrix2param_(matrix)

        modal = self.eig_parametrizer.modal_matrix
        # Note that the modal matrix is U(2); it should be mapped to SU(2)
        # before representing it using euler angles
        rdet = torch.linalg.det(modal).unsqueeze(-1).unsqueeze(-1).conj()**0.5
        # rdet: rooted determinant
        eul_param, eul_logJ = self.eul_parametrizer.matrix2param_(modal * rdet)

        param = torch.stack([eig_param[..., 0], *eul_param], dim=-1)
        return param, eig_logJ + eul_logJ

    def param2matrix_(self, param, reduce_=None):
        # reduce_ is just for legacy and is irrelevant here
        eig_param = param[..., 0].unsqueeze(-1)
        eul_param = torch.unbind(param[..., 1:], dim=-1)

        modal, eul_logJ = self.eul_parametrizer.param2matrix_(eul_param)
        self.eig_parametrizer.modal_matrix = modal

        matrix, eig_logJ = self.eig_parametrizer.param2matrix_(eig_param)

        return matrix, eul_logJ + eig_logJ


# =============================================================================
def sum_density(x):
    return torch.sum(x, dim=list(range(1, x.dim())))
