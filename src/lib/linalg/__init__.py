# Copyright (c) 2023 Javad Komijani

"""This module has extensions to the linalg packages in torch or numpy."""

from .qr_decomposition import haar_qr, haar_sqr
from .sv_decomposition import unique_svd
from .eig_decomposition import eigu  # eig for unitray matrices

from .euler_angles import su2_to_euler_angles
from .euler_angles import euler_angles_to_su2
