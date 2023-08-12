# Copyright (c) 2021-2023 Javad Komijani

"""This module has utilities to handle staples related to gauge links."""


import torch
from ..linalg import svd

mul = torch.matmul


# =============================================================================
class TemplateStaplesHandle:

    def __init__(self, staples=None):
        self.staples_svd = None if staples is None else svd(staples)

    def staple(self, link, staples=None):
        """
        Return "stapled links" or "effective projected plaquettes" that are
        defined as the links multiplied by SU(n) matrices obtained by applying
        SVD on the staples.
        If staples are not provided the fixed staples used when the class was
        instantiated will be used.
        """
        if staples is not None:
            self.staples_svd = svd(staples)

        svd_ = self.staples_svd
        if svd_ is None:
            raise Exception("staples are not defined!")

        eff_proj_plaq = link @ svd_.sUVh
        return eff_proj_plaq, (svd_.S, svd_.det_uvh.squeeze(-1))

    def unstaple(self, eff_proj_plaq, staples=None):
        # see self.push2links, which is used instead of unstaple

        if staples is not None:
            self.staples_svd = svd(staples)

        svd_ = self.staples_svd
        if svd_ is None:
            raise Exception("staples are not defined!")

        link = eff_proj_plaq @ svd_.sUVh.adjoint()
        return link

    def push2links(self, link, *, eff_proj_plaq_old, eff_proj_plaq_new):
        if eff_proj_plaq_old is None:
            return  eff_proj_plaq_new @ link
        else:
            return (eff_proj_plaq_new @ eff_proj_plaq_old.adjoint()) @ link

    def free_memory(self):
        self.staples_svd = None


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
