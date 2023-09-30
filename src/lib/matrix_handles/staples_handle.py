# Copyright (c) 2021-2023 Javad Komijani

"""This module has utilities to handle staples related to gauge links."""


import torch
from ..linalg import special_svd

mul = torch.matmul


# =============================================================================
class TemplateStaplesHandle:

    def __init__(self, onesided=False):
        self.onesided = onesided

    def staple(self, link, *, staples):
        """
        Return `slink` (stapled link) defined as `link` multiplied by SU(n)
        matrices that are obtained by performing SVD on the sum of the
        corresponding `staples.`
        """
        svd_ = special_svd(staples)
        if self.onesided:
            slink = link @ svd_.sU @ svd_.Vh  # slink stands for stapled link
        else:
            slink = svd_.Vh @ link @ svd_.sU
        return slink, svd_

    def unstaple(self, slink, svd_):
        """Invert the `staple` method.

        For pushing the changes in `slink` to corresponding `link` use the
        `push2link` method.
        """
        if self.onesided:
            link = slink @ (svd_.sU @ svd_.Vh).adjoint()
        else:
            link = svd_.Vh.adjoint() @ slink @ svd_.sU.adjoint()

        return link

    def push2link(self, link, *, slink_rotation, svd_):

        if not self.onesided:
            slink_rotation = svd_.Vh.adjoint() @ slink_rotation @ svd_.Vh

        return slink_rotation @ link


# =============================================================================
class FixedStaplesHandle:

    def __init__(self, staples):
        self.svd_ = special_svd(staples)
        self.suvh = svd_.sU @ svd_.Vh

    def staple(self, link):
        slink = link @ svd_.suvh  # slink stands for stapled link
        return slink, self.svd_

    def unstaple(self, slink):
        return slink @ self.suvh.adjoint()


# =============================================================================
class WilsonStaplesHandle(TemplateStaplesHandle):

    vector_axis = 1  # The batch axis is 0

    def makesure_correct_vector_axis(self, vector_axis):
        assert self.vector_axis == vector_axis, "vector axis?"

    @classmethod
    def calc_staples(cls, links, *, mu, nu_list):
        """Calculate the staples (from the Wilson gauge action) corresponding
        to the `links` that are in `mu` direction and summed over mu-nu planes
        with nu in `nu_list`.

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
        links : tensor
            Tensor of gauge links.
        mu : int
            Direction of the links with them the staples are associated.
        """
        return sum(
            [cls.calc_planar_staples(links, mu=mu, nu=nu) for nu in nu_list]
            )

    @classmethod
    def calc_planar_staples(
            cls, links, *, mu, nu, up_only=False, down_only=False
            ):
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

        if cls.vector_axis == 0:
            x_mu = links[mu]
            x_nu = links[nu]
        else:
            # then assume it is 1
            x_mu = links[:, mu]
            x_nu = links[:, nu]

        # U = x_mu  # we do not need U
        c = x_nu
        a = torch.roll(c, -1, dims=1 + mu)
        b = torch.roll(x_mu, -1, dims=1 + nu)

        if up_only:
            return cls.staple1_rule(a, b, c)

        e = torch.roll(x_mu, +1, dims=1 + nu)
        f = torch.roll(c, +1, dims=1 + nu)
        d = torch.roll(a, +1, dims=1 + nu)

        if down_only:
            return cls.staple2_rule(d, e, f)

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
        pass  # not ready

    def staple(self):
        pass  # not ready

    def unstaple(self):
        pass  # not ready

    @staticmethod
    def staple1_rule(a, b, c):
        return a * torch.conj(b * c)

    @staticmethod
    def staple2_rule(d, e, f):
        return torch.conj(e * d) * f
