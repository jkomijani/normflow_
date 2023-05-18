# Copyright (c) 2023 Javad Komijani

"""This module has extensions to the linalg packages in torch or numpy."""

import torch


def haar_qr(x, q_only=False):
    """Return a phase corrected version of qr decomposition that can be used to
    generate unitary matrices with the so-called haar measure.

    Performing the phase correction on q & r matrices, the diagonal terms of
    the r matrix are going to be real and positive.
    For further discussion see `[Mezzadri]`_.

    .. _[Mezzadri]:
        F. Mezzadri,
        "How to generate random matrices from the classical compact groups",
        :arXiv:`math-ph/0609050`.

    Parameters
    ----------
    x : tensor,
        the set of matrices for qr decomposition

    q_only : boolean (otpional)
        if True, only q will be returned rather than q & r (default is False).
    """
    q, r = torch.linalg.qr(x, mode='complete')
    # we now "correct" the phase of columns & rows in q & r, respectively
    d = torch.diagonal(r, dim1=-2, dim2=-1)
    phase = (d/torch.abs(d))
    q = q * phase.unsqueeze(-2)  # correct the phase of columns
    if q_only:
        return q
    r = r * (1 / phase).unsqueeze(-1)  # correct the phase of rows
    # Note that x = q @ r before & after the pahse correction
    return q, r


def haar_sqr(x, u1_matrix_rep=False):
    r"""Decompose x to SU(n), U(1) & invertible upper triangular matrices.

    Here 'sqr' stands for 'special qr'. First 'haar_qr' is called for qr
    decomposition of the input. Then q is decomposed to a special unitary
    matrix and a U(1) element. We can define the U(1) part to be just a complex
    number or an n times n matrix representation, like q and r; for the latter
    case, we define :math:`u = det(q)^{1/n} I_{n\times n}`.


    Parameters
    ----------
    x : tensor,
        the set of matrices for special qr decomposition

    u1_matrix_rep : boolean (otpional)
        returns a complex/matrix representation for the u(1) element if
        False/True (defualt is False).
    """
    q, r = haar_qr(x)
    det = torch.linalg.det(q)
    # Remarks:
    # 1. det \in U(1)
    # 2. det^(1/n) covers only 1/n-th of U(1) -> volume 1/n-th of U(1)
    # 3. determinant of det^(1/n) I_{n x n} covers U(1); stretching fac. n
    n = x.shape[-1]
    phase = torch.pow(det, -1./n).unsqueeze(-1).unsqueeze(-1)

    if u1_matrix_rep:
        u = torch.diag_embed(torch.repeat_interleave(1 / phase.squeeze(-1), n, dim=-1))
    else:
        u = det
    return q * phase, u, r
