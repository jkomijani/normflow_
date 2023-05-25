# Copyright (c) 2021-2023 Javad Komijani

"""This module has utilities to handle staples related to gauge links."""


import torch

mul = torch.matmul


class TemplateStaplesHandle:

    def __init__(self, staples=None):

        self.staples_svd, self.staples_svd_phasor = self._svd(staples)

    def staple(self, x, staples=None):
        if staples is not None:
            self.staples_svd, self.staples_svd_phasor = self._svd(staples)

        svd = self.staples_svd
        if svd is not None:
            x = (svd.Vh @ x @ svd.U) * self.staples_svd_phasor.conj()

        return x

    def unstaple(self, x, staples=None):
        if staples is not None:
            self.staples_svd, self.staples_svd_phasor = self._svd(staples)

        svd = self.staples_svd
        if svd is not None:
            x = (svd.Vh.adjoint() @ x @ svd.U.adjoint()) * self.staples_svd_phasor

        return x

    @staticmethod
    def _svd(z):
        if z is None:
            svd, svd_phasor = None, None
        else:
            svd = torch.linalg.svd(z)
            svd_phasor = torch.linalg.det(z)**(1/z.shape[-1])
            svd_phasor = svd_phasor.reshape(*z.shape[:-2], 1, 1)
        return svd, svd_phasor


class PlanarStaplesHandle:

    def calc_wilson_staples(self, x_mu, x_nu, zpmask):
        """Calculate and return the staples from the Wilson gauge action.
        
        Stables of the Wilson gauge action are shown in the following cartoon:

            >>>     --b--
            >>>    c|   |a
            >>>     @ U @    +   @ U @
            >>>                 f|   |d
            >>>                  --e--

        where `@ U @` shows the central link for which the staples are going to
        be calculated according to a zebra planar mask,
        where the plane is specified with indices `(mu, nu)` and `parity`
        specified in the zebra planar mask.

        In the following cartoon:
        1. Horizontal & vertical axes are `(mu, nu)` axes.
        2. Rows with `;` inside plaquettes are called black and the rest white.
        3. If `parity == 1`, the black and white rows turn to the other color.
        4. `x_nu_white` are the links in `nu` direction of white plaquettes.
        5. `x_mu_white` are the lower links in `mu` direction of white plaquettes.
        6. `x_mu_black` are the lower links in `mu` direction of black plaquettes.


            >>>    |;|;|;|;|;|;|
            >>>    -------------
            >>>    | | | | | | |
            >>>    -------------
            >>>    |;|;|;|;|;|;|
            >>>    -------------
            >>>    | | | | | | |
            >>>    -------------   -> mu

        Parameters
        ----------
        x_mu, x_nu : tensor, tensor
            The links in mu and nu directions.
        zpmask : mask
            A zebra planar mask that is supposed to have mu, nu, and parity
            properties and a split method.
        """
        x_mu_white, x_mu_black = zpmask.split(x_mu)
        x_nu_white, x_nu_black = zpmask.split(x_nu)

        mu, nu, parity = zpmask.mu, zpmask.nu, zpmask.parity

        # Calculate the staples $a 1/b 1/c$  &  $1/d 1/e f$
        #   --b--
        #  c|   |a
        #   @ U @    +    @ U @
        #                f|   |d
        #                 --e--

        # U = x_mu_white  # we do not need U
        c = x_nu_white
        f = x_nu_black if parity == 1 else torch.roll(x_nu_black, 1, dims=1 + nu)
        a = torch.roll(a, -1, dims=1 + mu)
        d = torch.roll(f, -1, dims=1 + mu)
        b = x_mu_black if parity == 0 else torch.roll(x_mu_black, -1, dims=1 + nu)
        e = torch.roll(b, 1, dims=1 + nu)  # Stride of b (x_mu_black) is 2

        return self.staple1_rule(a, b, c) + self.staple2_rule(d, e, f)

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


class U1PlanarStaplesHandle(PlanarStaplesHandle):
    """Properties and methods are chosen to be consistent with SU(n)."""

    @staticmethod
    def staple1_rule(a, b, c):
        return a * torch.conj(b * c)

    @staticmethod
    def staple2_rule(d, e, f):
        return torch.conj(e * d) * f
