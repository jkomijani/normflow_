# Copyright (c) 2023 Javad Komijani

"""This module has extensions to the linalg packages in torch or numpy."""

import torch


# =============================================================================
def su2_to_euler_angles(matrix, alt_param=False):
    r"""Perform Euler decomposition of SU(2) matrices and return the angles.

    Here we follow [Ref](https://ncatlab.org/nlab/show/Euler+angle);
    also see (3.3.21) of Sakurai's book with a slightly different
    paramerization.

    This function can be called with `alt_param` option set to False (default)
    or True:

        >>> [phi, theta, psi] = su2_to_euler_angles(matrix)
        >>> [sum_, abs00, diff] = su2_to_euler_angles(matrix, alt_param=True)

    where sum_ is phi + psi, diff is phi - psi, and abs00 is cos(theta/2),
    which is always positive.

    Parameters
    ----------
    matrix : tensor
        the matrix to be mapped to Euler angles.

    alt_param : bool (optional)
        if True, returns an alt_param parametrization as explained above
        (default is False).
    """
    abs00 = torch.abs(matrix[..., 0, 0])  # \in [0, 1]
    angle00 = torch.angle(matrix[..., 0, 0])  # \in (-pi, pi]
    angle01 = torch.angle(-1j * matrix[..., 0, 1])  # \in [-pi, pi]

    if alt_param:
        return [angle00, abs00, angle01]
    else:
        phi = angle00 + angle01  # \in (-2 pi, 2 pi]
        psi = angle00 - angle01  # \in (-2 pi, 2 pi]
        theta = 2 * torch.acos(abs00)
        return [phi, theta, psi]


# =============================================================================
def euler_angles_to_su2(phi, theta, psi, alt_param=False):
    """Performing the opposite of su2_to_euler_angles.

    For details see su2_to_euler_angles.

    Parameters
    ----------
    phi, theta, psi : tensors
        Euler decomposed angles (if alt_param is False).

    alt_param : bool (optional)
        if True, [phi, theta, psi] should be interpreted as
        [(phi+psi)/2, cos(theta/2), (phi - psi)/2] (default is False).
    """
    if alt_param:
        angle00, abs00, angle01 = phi, theta, psi
    else:
        angle00 = (phi + psi) / 2
        angle01 = (phi - psi) / 2
        abs00 = torch.cos(theta / 2)

    m00 = abs00 * torch.exp(1j * angle00)
    m01 = 1j * torch.sqrt(1 - abs00**2) * torch.exp(1j * angle01)

    matrix = torch.stack([m00, m01, -m01.conj(), m00.conj()], dim=-1)
    return matrix.reshape(*m00.shape, 2, 2)
