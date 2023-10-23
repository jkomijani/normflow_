# Copyright (c) 2023 Javad Komijani


import torch

from ..scalar.couplings_ import Coupling_
from .gauge_param_couplings_ import Pade11Coupling_, Pade22Coupling_
from .gauge_param_couplings_ import SU2RQSplineCoupling_, SU3RQSplineCoupling_


# =============================================================================
class DualCoupling_(Coupling_):

    def forward(self, x, s, log0=0):
        x_active, x_invisible = self.mask.split(x)
        s_frozen, s_invisible = self.mask.split(s)

        x_active, logJ = self.atomic_forward(
            x_active=x_active, x_frozen=s_frozen, parity=0, net=self.nets[0]
            )

        return self.mask.cat(x_active, x_invisible), log0 + logJ

    def backward(self, x, s, log0=0):
        x_active, x_invisible = self.mask.split(x)
        s_frozen, s_invisible = self.mask.split(s)

        x_active, logJ = self.atomic_backward(
            x_active=x_active, x_frozen=s_frozen, parity=0, net=self.nets[0]
            )

        return self.mask.cat(x_active, x_invisible), log0 + logJ


# =============================================================================
class Pade11DualCoupling_(DualCoupling_, Pade11Coupling_):
    pass


class Pade22DualCoupling_(DualCoupling_, Pade22Coupling_):
    pass


class SU2RQSplineDualCoupling_(DualCoupling_, SU2RQSplineCoupling_):
    pass


class SU3RQSplineDualCoupling_(DualCoupling_, SU3RQSplineCoupling_):
    pass
