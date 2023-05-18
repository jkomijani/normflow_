# Copyright (c) 2023 Javad Komijani

"""This module has extensions to the linalg packages in torch or numpy."""

import torch


def sgn_qr(x, q_only=False):
    """Return a "sign-corrected" version of qr decomposition.

    This function can be used to generate unitary matrices as explained in
    `[Mezzadri]`_.

    .. _[Mezzadri]:
        F. Mezzadri,
        "How to generate random matrices from the classical compact groups",
        :arXiv:`math-ph/0609050`.
    """
    q, r = torch.linalg.qr(x, mode='complete')
    # we now "correct" the signs of columns & rows in q & r, respectively
    d = torch.diagonal(r, dim1=-2, dim2=-1)
    sgn = (d/torch.abs(d))
    q *= sgn.unsqueeze(-2)  # correct the signs of columns
    if q_only:
        return q
    r *= sgn.unsqueeze(-1)  # correct the signs of rows
    # Note that x = q @ r before & after correcting for the sign
    return q, r
