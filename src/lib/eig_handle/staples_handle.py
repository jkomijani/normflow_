# Copyright (c) 2021-2023 Javad Komijani

"""This module has utilities to handle staples related to gauge links."""


import torch
from ..linalg import unique_svd

mul = torch.matmul


# =============================================================================
class TemplateStaplesHandle:

    def __init__(self, staples=None, sandwich=False):
        self.sandwich = sandwich
        self.staples_svd, self.staples_svd_phasor = self._svd(staples)

    def staple(self, link, staples=None):

        if staples is not None:
            self.staples_svd, self.staples_svd_phasor = self._svd(staples)

        svd = self.staples_svd
        if svd is not None:
            phasor = self.staples_svd_phasor
            if self.sandwich:
                eff_proj_plaq = (svd.Vh @ link @ svd.U) * phasor.conj()
            else:
                eff_proj_plaq = link @ (svd.U @ svd.Vh) * phasor.conj()
            phasor = phasor.squeeze(-1)
            s_and_phase = torch.cat([svd.S, phasor.real, phasor.imag], -1)
        else:
            eff_proj_plaq = x
            s_and_phase = svd.S

        return eff_proj_plaq, s_and_phase

    def unstaple(self, eff_proj_plaq, staples=None):

        if staples is not None:
            self.staples_svd, self.staples_svd_phasor = self._svd(staples)

        svd = self.staples_svd
        if svd is not None:
            phasor = self.staples_svd_phasor
            if self.sandwich:
                x = (svd.Vh.adjoint() @ eff_proj_plaq @ svd.U.adjoint()) * phasor
            else:
                x = eff_proj_plaq @ (svd.U @ svd.Vh).adjoint() * phasor
        else:
            x = eff_proj_plaq

        return x

    def push2links(self, x, *, eff_proj_plaq_old, eff_proj_plaq_new):
        return (eff_proj_plaq_new @ eff_proj_plaq_old.adjoint()) @ x

    @staticmethod
    def _svd(z):
        if z is None:
            svd, svd_phasor = None, None
        else:
            # svd = torch.linalg.svd(z)
            svd = unique_svd(z)  # torch.linalg.svd(z)
            det_z = torch.linalg.det(z)
            svd_phasor = (det_z / torch.abs(det_z))**(1 / z.shape[-1])
            svd_phasor = svd_phasor.reshape(*z.shape[:-2], 1, 1)
        return svd, svd_phasor

    def free_memory(self):
        self.staples_svd, self.staples_svd_phasor = None, None


# =============================================================================
class WilsonStaplesHandle(TemplateStaplesHandle):

    @classmethod
    def calc_staples(cls, cfgs, *, mu, nu_list):
        """Calculate and return the staples from the Wilson gauge action.

        Stables of the Wilson gauge action in any plane are shown in the
        following cartoon:

            >>>     --b--
            >>>    c|   |a
            >>>     @ U @    +   @ U @
            >>>                 f|   |d
            >>>                  --e--

        where `@ U @` shows the central link for which the staples are going to
        be calculated.

        Parameters
        ----------
        cfgs : tensor
            Tensor of configurations.
        mu : int
            Direction of the links with them the staples are associated.
        """
        return sum(
            [cls.calc_planar_staples(cfgs, mu=mu, nu=nu) for nu in nu_list]
            )

    @classmethod
    def calc_planar_staples(cls, cfgs, *, mu, nu, up_only=False):
        """Similar to calc_staples, except that the staples are calculated on
        mu-nu plane.
        """
        # In the plane specified with mu and nu, calculate the staples
        # $a 1/b 1/c$ and $1/d 1/e f$
        #
        #   --b--
        #  c|   |a
        #   @ U @    +    @ U @
        #                f|   |d
        #                 --e--
        x_mu = cfgs[:, mu]
        x_nu = cfgs[:, nu]

        # U = x_mu  # we do not need U
        c = x_nu
        a = torch.roll(c, -1, dims=1 + mu)
        b = torch.roll(x_mu, -1, dims=1 + nu)

        if up_only:
            return cls.staple1_rule(a, b, c)

        e = torch.roll(x_mu, +1, dims=1 + nu)
        f = torch.roll(c, +1, dims=1 + nu)
        d = torch.roll(a, +1, dims=1 + nu)

        return cls.staple1_rule(a, b, c) + cls.staple2_rule(d, e, f)

    @staticmethod
    def staple1_rule(a, b, c):
        """return :math:`a  b^\dagger  c^\dagger`."""
        #   --b--
        #  c|   |a
        #   @ U @
        return mul(a, mul(c, b).adjoint())

    @staticmethod
    def staple2_rule(d, e, f):
        """return :math:`d^\dagger  e^\dagger  f`."""
        #   @ U @
        #  f|   |d
        #   --e--
        return mul(mul(e, d).adjoint(), f)


# =============================================================================
class U1WilsonStaplesHandle(WilsonStaplesHandle):
    """Properties and methods are chosen to be consistent with SU(n)."""

    def __init__(self):
        pass

    def staple(self):
        pass

    def unstaple(self):
        pass

    @staticmethod
    def staple1_rule(a, b, c):
        return a * torch.conj(b * c)

    @staticmethod
    def staple2_rule(d, e, f):
        return torch.conj(e * d) * f
