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


def append_suvh(svd):
    """Return a new svd object that includes U V^\dagger"""
    uvh = svd.U @ svd.Vh
    rdet = torch.det(uvh)**(1 / uvh.shape[-1])  # root of determinant
    # We now make determinant of uvh unity:
    uvh = uvh / rdet.reshape(*rdet.shape, 1, 1)
    return AttributeDict4SVD(U=svd.U, S=svd.S, Vh=svd.Vh, rdet_uvh=rdet, sUVh=uvh)
