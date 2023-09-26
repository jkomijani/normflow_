# Copyright (c) 2023 Javad Komijani

"""This module has extensions to the linalg packages in torch or numpy."""


import torch


try:
    from torch_linalg_ext import svd, eigh, eigu

except:
    print("Warning: Could not locate torch_linalg_ext; uses torch.linalg")
    from torch.linalg import svd, eigh, eig as eigu


from .qr_decomposition import haar_qr, haar_sqr

from .euler_angles import su2_to_euler_angles
from .euler_angles import euler_angles_to_su2

from .mean import neighbor_mean


class AttributeDict4SVD:
    """For accessing a dict key like an attribute."""

    def __init__(self, **dict_):
        self.__dict__.update(**dict_)

    def __repr__(self):
        str_ = "svd:\n"
        for key, value in self.__dict__.items():
            str_ += f"{key}={value}\n"
        return str_


def special_svd(matrix):
    """Return a new svd object, in which U is scaled by a phase, and called sU,
    such that sU @ Vh is special unitary
    """
    svd_ = svd(matrix)
    rdet_angle = torch.angle(torch.det(matrix)) / svd_.U.shape[-1]  # r: rooted
    s_u = svd_.U * torch.exp(-1j * det_angle.reshape(*rdet_angle.shape, 1, 1))
    return AttributeDict4SVD(
        U=svd.U, S=svd.S, Vh=svd.Vh, rdet_angle=rdet_anlge, sU=s_u
        )
