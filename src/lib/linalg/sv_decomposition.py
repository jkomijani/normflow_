# Copyright (c) 2023 Javad Komijani

"""This module has extensions to the linalg packages in torch or numpy."""

import torch


class AttributeDict:
    """For accessing a dict key like an attribute."""

    def __init__(self, **dict_):
        self.__dict__.update(**dict_)

    def __repr__(self):
        return str(self.__dict__)


def unique_svd(x):
    """Return a modified version of SVD in which the degrees of redundancy are
    fixed.
    """
    svd = torch.linalg.svd(x)
    # we now "fix" the phase of columns & rows in U & Vh, respectively
    d = torch.diagonal(svd.U, dim1=-2, dim2=-1)
    phase = d / torch.abs(d)
    u = svd.U * (1 / phase).unsqueeze(-2)  # correct the phase of columns
    vh = svd.Vh * phase.unsqueeze(-1)  # correct the phase of rows
    s = svd.S.clone()
    return AttributeDict(U=u, S=s, Vh=vh)


def svd_2x2(matrix):
    """Use a closed form expression to calculate SVD of a 2x2 matrix."""

    s_sq, v = eigh_2x2(matrix.adjoint() @ matrix)
    s = torch.sqrt(s_sq)
    u = matrix @ v @ (torch.diag_embed(1 / s) + 0j)
    return AttributeDict(U=u, S=s, Vh=v.adjoint())


def eigh_2x2(matrix):
    """Use a closed form expression to calculate eigenvalues and eigenvectors
    of a 2x2 hermitian matrix.
    """
    # For the algorithm e.g. see,
    # https://hal.science/hal-01501221v1/preview/matrix_exp_and_log_formula.pdf
    # but note that we construct the eigenvectors slighly differently

    # We assume matrix is hermition, if not we treat matrix similar to pytorch
    m00 = torch.real(matrix[..., 0, 0])
    m10 = matrix[..., 1, 0]
    m01 = m10.conj()
    m11 = torch.real(matrix[..., 1, 1])
    trace = m00 + m11
    dis = torch.sqrt((m00 - m11)**2 + 4 * (m10.real**2 + m10.imag**2))  # discriminant

    eigval = torch.stack([(trace - dis) / 2, (trace + dis) / 2], dim=-1)

    eigvec = torch.zeros_like(matrix)
    eigvec[..., 0, 0] = 1.
    eigvec[..., 1, 1] = 1.

    cond_ = (m10.real**2 + m10.imag**2) > 1e-30

    eigvec[..., 0, 0][cond_] = (eigval[..., 0] - m11)[cond_] + 0j
    eigvec[..., 0, 1][cond_] = m01[cond_]
    eigvec[..., 1, 0][cond_] = m10[cond_]
    eigvec[..., 1, 1][cond_] = (eigval[..., 1] - m00)[cond_] + 0j

    eigvec = eigvec / torch.linalg.vector_norm(eigvec, dim=-2, keepdim=True)

    return eigval, eigvec
