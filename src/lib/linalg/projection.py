# Copyright (c) 2023 Javad Komijani

"""This module has extensions to the linalg packages in torch or numpy."""

import torch


def project_su2(matrix):
    """projects the current ``2x2`` complex matrix ``Q`` onto SU(2) matrix `W``
    by maximizing ``Re Tr (Q^\dagger W)``.
    """

    # The SU(2) matrix is represented as v0 + i * Sum_j (sigma_j * vj)
    #     
    #     | A+iB     C+iD |     
    # M = |               |
    #     | E+iF     G+iH | 
    #     
    #   =  1/2*[ (A+G)*I + i*(B-H)*sigma_z + i*(F+D)*sigma_x + i*(C-E)*sigma_y ]
    #    + i/2*[ (B+H)*I - i*(A-G)*sigma_z - i*(C+E)*sigma_x - i*(F-D)*sigma_y ]
    #     
    # The second line does not contribute to `Re Tr (Q^\dagger W)`, so we drop it.
    # When the first line is identical to zero, we simply assign the identity matrix
    # as the projection of M onto SU(2).

    v0 = matrix[..., 0, 0].real + matrix[..., 1, 1].real
    v3 = matrix[..., 0, 0].imag - matrix[..., 1, 1].imag
    v1 = matrix[..., 0, 1].imag + matrix[..., 1, 0].imag
    v2 = matrix[..., 0, 1].real - matrix[..., 1, 0].real

    v_sq = v0**2 + v1**2 + v2**2 + v3**2

    r = 1 / v_sq**0.5    # for normalization

    out_mat = torch.zeros_like(matrix)

    out_mat[..., 0, 0] = (v0 - v3*1J) * r
    out_mat[..., 0, 1] = (-v2 - v1*1J) * r
    out_mat[..., 1, 0] = (v2 - v1*1J) * r
    out_mat[..., 1, 1] = (v0 + v3*1J) * r

    return out_mat
