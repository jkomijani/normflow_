# Copyright (c) 2021-2022 Javad Komijani

"""This module has utilities to handle plaquettes and how they are related to
links.
"""


import torch

mul = torch.matmul


class PlaqHandle:

    def calc_zpmasked_open_plaq(self, x_mu, x_nu, zpmask):
        # zpmasked -> zebra planar masked
        """Calculate un-traced plaquettes according to a zebra planar mask,
        where the plane is specified with indices `(mu, nu)` and `parity`
        specified in the zebra planar mask.

        In the following cartoon:
        1. Horizontal & vertical axes are `(mu, nu)` axes.
        2. Rows with `;` inside plaquettes are called black and the rest white.
        3. If `parity == 1`, the black and white rows turn to the other color.
        4. `x_nu_white` are the links in `nu` direction of white plaquettes.
        5. `x_mu_white` are the lower links in `mu` direction of white plaquettes.
        6. `x_mu_black` are the lower links in `mu` direction of black plaquettes.

        |;|;|;|;|;|;|
        -------------
        | | | | | | |
        -------------
        |;|;|;|;|;|;|
        -------------
        | | | | | | |
        -------------   -> mu

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

        # Calculate the open plaq $a b 1/c 1/d$
        #    -c-
        #  d|   |b
        #    -a-
        a = x_mu_white
        b = torch.roll(x_nu_white, -1, dims=1 + mu)
        c = x_mu_black if parity == 0 else torch.roll(x_mu_black, -1, dims=1 + nu)
        d = x_nu_white

        return self.plaq_rule(a, b, c, d)

    def calc_zpmasked_open_plaqlongplaq(self, x_mu, x_nu, zpmask):
        """Similar to ``calc_zpmasked_open_plaq()`` but also calculate the
        so-called long plaquettes (long in the nu direction).

        The output will have a channels axis (at dim=1), where the first and
        second channels are the open plaquettes and long plaquettes,
        respectively. However, note that the corresponding channels axis can
        change to 2 if another channels axis is added to the data. For example,
        when we canculate the eigenvalues of the open plaquettes, we may put
        the eigenvalues as a new channels axis (at dim=1), which in turn, moves
        the axis corresponding to the open plaquettes and long plaquettes to 2.
        """
        # zpmasked -> zebra planar masked
        x_mu_white, x_mu_black = zpmask.split(x_mu)
        x_nu_white, x_nu_black = zpmask.split(x_nu)

        mu, nu, parity = zpmask.mu, zpmask.nu, zpmask.parity

        # the output will have two channels: open plaq a long open plaq
        out_shape = (x_mu_white.shape[0], 2, *x_mu_white.shape[1:])
        out = torch.zeros_like(x_mu).reshape(*out_shape)

        # First, calculate the open plaq $a b 1/c 1/d$
        #    -c-
        #  d|   |b
        #    -a-
        a = x_mu_white
        b = torch.roll(x_nu_white, -1, dims=1 + mu)
        c = x_mu_black if parity == 0 else torch.roll(x_mu_black, -1, dims=1 + nu)
        d = x_nu_white

        out[:, 0] = self.plaq_rule(a, b, c, d)

        # Second, calculate the long open plaq $a b 1/c 1/d 1/e f$
        #    -b-
        #  a|   |c
        #    ---
        #  f|   |d
        #    -e-
        a, b, c = d, c, b
        f = x_nu_black if parity == 1 else torch.roll(x_nu_black, 1, dims=1 + nu)
        d = torch.roll(f, -1, dims=1 + mu)
        e = torch.roll(b, 1, dims=1 + nu)  # Stride of b (x_mu_black) is 2

        out[:, 1] = self.longplaq_rule(a, b, c, d, e, f)

        return out

    def push_plaq2links(self, *, new_plaq, old_plaq, links, zpmask):
        x_mu = links
        x_mu_white, x_mu_black = zpmask.split(x_mu)
        x_mu_white = self.update_link_rule(new_plaq, old_plaq, x_mu_white)
        x_mu = zpmask.cat(x_mu_white, x_mu_black)
        return x_mu

    @staticmethod
    def plaq_rule(a, b, c, d):
        """return :math:`a  b  c^\dagger  d^\dagger`."""
        #    -c-
        #  d|   |b
        #    -a-
        return mul(mul(a, b), mul(d, c).adjoint())

    @staticmethod
    def longplaq_rule(a, b, c, d, e, f):
        """return :math:`a  b  c^\dagger d^\dagger e^\dagger f`."""
        #    -b-
        #  a|   |c
        #    ---
        #  f|   |d
        #    -e-
        return mul(mul(mul(a, b), mul(e, mul(d, c)).adjoint()), f)

    @staticmethod
    def update_link_rule(a, b, c):
        if b is None:
            return mul(a, c)
        else:
             return mul(mul(a, b.adjoint()), c)


class LongPlaqHandle(PlaqHandle):

    def calc_zpmasked_open_plaq(self, *args):
        return super().calc_zpmasked_open_plaqlongplaq(*args)

    def push_plaq2links(self, *, new_plaq, old_plaq, **kwargs):
        return super().push_plaq2links(
                            new_plaq=new_plaq[:, 0],
                            old_plaq=None if old_plaq is None else old_plaq[:, 0],
                            **kwargs
                            )


class U1PlaqHandle(PlaqHandle):
    """Properties and methods are chosen to be consistent with SU(n)."""

    @staticmethod
    def plaq_rule(a, b, c, d):
        return a * b * torch.conj(d * c)

    @staticmethod
    def longplaq_rule(a, b, c, d, e, f):
        return a * b * torch.conj(e * d * c) * f

    @staticmethod
    def update_link_rule(a, b, c):
        if b is None:
            return a * c
        else:
            return a * b.conj() * c


class U1LongPlaqHandle(U1PlaqHandle, LongPlaqHandle):
    pass


SUnPlaqHandle = PlaqHandle
SUnLongPlaqHandle = LongPlaqHandle
