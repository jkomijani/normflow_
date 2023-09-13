# Copyright (c) 2021-2022 Javad Komijani

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
        >>> [sum_, abs00, diff] = su2_to_euler_angles(matrix, alt_param=True)

    where sum_ is phi + psi, diff is phi - psi, and abs00 is cos(theta/2),
    which is always positive.

    Inside this class we always set alt_param to True.
    """

    def __init__(self, nonfixed_angles_list=[1]):
        self.nonfixed_angles_list = nonfixed_angles_list
        self.euler_angles = [None, None, None]

    def matrix2angles_(self, matrix):
        """Calculate, save (for later use), and return the euler angles of
        the input matrix along with the Jacabian of the transformation.
        """
        self.euler_angles = su2_to_euler_angles(matrix, alt_param=True)
        logJ = sum_density(
                self.calc_log_jacobian(self.euler_angles, alt_param=True)
                )  # up to a constant
        return self.euler_angles, logJ

    def angles2matrix_(self, euler_angles):
        """Reverse of matrix2angles_."""
        matrix = euler_angles_to_su2(*euler_angles, alt_param=True)
        logJ = -sum_density(
                self.calc_log_jacobian(euler_angles, alt_param=True)
                )  # up to an additive constant
        return matrix, logJ

    def matrix2param_(self, matrix):
        """Return the euler angles normalized so that all are in [0, 1].
        
        The scaling of the euler angles is not incorporated to logJ because it
        is going to be undone when param2matrix_ is called.
        """
        euler_angles, logJ = self.matrix2angles_(matrix)
        euler_param = [None for _ in self.nonfixed_angles_list]
        for l, k in enumerate(self.nonfixed_angles_list):
            if k==1:  # already \in [0, 1]
                euler_param[l] = euler_angles[k]
            else:
                euler_param[l] = euler_angles[k] / (2 * pi) + 0.5
        return euler_param, logJ

    def param2matrix_(self, euler_param):
        """Reverse of matrix2param_."""
        euler_angles = [angle for angle in self.euler_angles]
        for l, k in enumerate(self.nonfixed_angles_list):
            if k==1:  # we used alt_param = True below
                euler_angles[k] = euler_param[l]
            else:
                euler_angles[k] = (euler_param[l] - 0.5) * (2 * pi)
        return self.angles2matrix_(euler_angles)

    @staticmethod
    def calc_log_jacobian(euler_angles, alt_param=True):
        """Return log of jacobian of transformation to euler angles up to an
        additive constant.
        """
        # *inverse* of Jacobian equals the volume of conjugacy class
        if alt_param:
            abs00 = euler_angles[1]
            return -torch.log(abs00)
        else:
            theta = euler_angles[1]
            return -torch.log(torch.sin(theta))

    @staticmethod
    def calc_jacobian(euler_angles, alt_param=True):
        """Return jacobian of transformation to euler angles up to a
        multiplicative constant.
        """
        if alt_param:
            abs00 = euler_angles[1]
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
