# Copyright (c) 2023 Javad Komijani

"""This module has extensions to the linalg packages in torch or numpy."""

try:
    from torch_linalg_ext import svd, eigh, eigsu

except:
    print("Warning: Could not locate torch_linalg_ext; uses torch.linalg")
    from torch.linalg import svd, eigh, eig as eigsu


from .qr_decomposition import haar_qr, haar_sqr

from .euler_angles import su2_to_euler_angles
from .euler_angles import euler_angles_to_su2
