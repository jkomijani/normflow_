# Copyright (c) 2023 Javad Komijani

"""This module has extensions to the linalg packages in torch or numpy."""

import torch


def eigu(x):
    """Return eigenvalues and eigenvectors of unitary matrices.

    The implementation is with torch.linalg.eigh, which is for hermitian
    matrices. Using torch.linalg.eig instead of `eigu` seems to accumulate
    error with large number of layers.

    We use

    .. math:

        U \Omega = \Omega \Lambda
        U^\dagger \Omega = \Omega \Lambda^\dagger

    to write

    .. math:

       (U + U^\dagger) \Omega = \Omega (\Lambda + \Lambda^\dagger)
       (U - U^\dagger) \Omega = \Omega (\Lambda - \Lambda^\dagger)

    to obtain eigenvalues and eigencetors of unitary matrices.

    Warning: The algorithm used here can lead to wrong decomposition when
    :math:`\sin(\lambda_i)` are degenerate, but with random matrices this is
    unlikely to happen.
    """
    eig_2sin, modal_matrix = torch.linalg.eigh(1J * (x.adjoint() - x))
    eig_2cos = torch.diagonal(
            modal_matrix.adjoint() @ (x.adjoint() + x) @ modal_matrix,
            dim1=-1, dim2=-2
            )
    eig = (eig_2cos + eig_2sin * 1J) / 2
    return eig, modal_matrix
