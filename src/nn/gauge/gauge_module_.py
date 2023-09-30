# Copyright (c) 2021-2023 Javad Komijani

"""This module contains new neural networks for transforming gauge fields.

The classes defined here are children of MatrixModule_ (and in turn Module_),
and the trailing underscore implies that the associated forward and backward
methods handle the Jacobians of the transformation.
"""


import torch

from .._core import ModuleList_
from ..matrix.matrix_module_ import MatrixModule_
from ..matrix.stapled_matrix_module_ import StapledMatrixModule_


# =============================================================================
class GaugeModuleList_(ModuleList_):

    def forward(self, x, log0=0):
        return super().forward(x.clone(), log0)

    def backward(self, x, log0=0):
        return super().backward(x.clone(), log0)

    def hack(self, x, log0=0):
        return super().hack(x.clone(), log0)


# =============================================================================
class GaugeModule_(MatrixModule_):
    """
    Parameters
    ----------
    net_: instance of Module_ or ModuleList_
        a net_ that takes as inputs a list that includes the svd-stapled-links
        that are going to be changed (the active ones) and the singular values
        of the corresponding staples.

    mu : int
        specifies the direction of links that are going to be changed
    
    nu_list : list of int
        (in combination w/ mu) specifies the plane of staples to be calculated
    
    staple_handle: class instance
         A class instance to handle matrices as expected in `self._kernel`

    matrix_handle: class instance
         A class instance to handle matrices as expected in `self._kernel`
    """

    def __init__(self, net_,
            *, mu, nu_list, staples_handle, matrix_handle, label="gauge_"
            ):
        super().__init__(net_, matrix_handle=matrix_handle)
        self.mu = mu
        self.nu_list = nu_list
        self.staples_handle = staples_handle
        self.label = label

    def forward(self, x, log0=0):
        x_mu = x[:, self.mu]

        staples = self.staples_handle.calc_staples(
                x, mu=self.mu, nu_list=self.nu_list
                )
        # slink: stapled link
        slink, svd_ = self.staples_handle.staple(x_mu, staples=staples)

        slink_rotation, logJ = super().forward(slink, log0, reduce_=True)

        x_mu = self.staples_handle.push2link(
                x_mu, slink_rotation=slink_rotation, svd_=svd_
                )

        x[:, self.mu] = x_mu

        return x, logJ

    def backward(self, x, log0=0):
        x_mu = x[:, self.mu]

        staples = self.staples_handle.calc_staples(
                x, mu=self.mu, nu_list=self.nu_list
                )
        slink, svd_ = self.staples_handle.staple(x_mu, staples=staples)

        slink_rotation, logJ = super().backward(slink, log0, reduce_=True)

        x_mu = self.staples_handle.push2link(
                x_mu, slink_rotation=slink_rotation, svd_=svd_
                )

        x[:, self.mu] = x_mu

        return x, logJ

    def _hack(self, x, log0=0, forward=True):
        """Similar to the forward method, but returns intermediate parts."""

        x = x.clone()

        x_mu = x[:, self.mu]

        staples = self.staples_handle.calc_staples(
                x, mu=self.mu, nu_list=self.nu_list
                )
        stack = [(x_mu, staples)]
        slink, svd_ = self.staples_handle.staple(x_mu, staples=staples)
        stack.append((slink, svd_))

        if forward:
            slink_rotation, logJ = super().forward(slink, log0, reduce_=True)
        else:
            slink_rotation, logJ = super().backward(slink, log0, reduce_=True)

        stack.append(super()._hack(slink, forward=forward, reduce_=True))
        stack.append((slink_rotation,))

        x_mu = self.staples_handle.push2link(
                x_mu, slink_rotation=slink_rotation, svd_=svd_
                )
        stack.append((x_mu,))

        x[:, self.mu] = x_mu
        stack.append((x, logJ))

        return stack

    def transfer(self, **kwargs):
        return self.__class__(
                self.net_.transfer(**kwargs),
                mu=self.mu,
                nu_list=self.nu_list,
                staples_handle=self.staples_handle,
                matrix_handle=self.matrix_handle,
                label=self.label
                )


# =============================================================================
class SVDGaugeModule_(StapledMatrixModule_):

    #   *** NOT UPDATED ***

    """
    Similar to GaugeModule_ but used singular values of the staples for
    processing.

    Parameters
    ----------
    net_: instance of Module_ or ModuleList_
        a net_ that takes as inputs a list that includes the svd-stapled-links
        that are going to be changed (the active ones) and the singular values
        of the corresponding staples.

    mu : int
        specifies the direction of links that are going to be changed
    
    nu_list : list of int
        (in combination w/ mu) specifies the plane of staples to be calculated
    
    mask : instance of Mask
        used for partitioning the links and their corresponding staples to
        active and frozen. Note that mask should be instantiated with e.g.
        split_form='directional_even-odd'.

    staple_handle: class instance
         A class instance to handle matrices as expected in `self._kernel`

    matrix_handle: class instance
         A class instance to handle matrices as expected in `self._kernel`
    """

    def __init__(self, net_,
            *, mu, nu_list, staples_handle, matrix_handle, mask, label="gauge_"
            ):
        super().__init__(net_, matrix_handle=matrix_handle, mask=mask)
        self.mu = mu
        self.nu_list = nu_list
        self.staples_handle = staples_handle
        self.label = label

    def forward(self, x, log0=0):
        x_mu = x[:, self.mu]

        staples = self.staples_handle.calc_staples(
                x, mu=self.mu, nu_list=self.nu_list
                )
        # below, sv stands for singular values (corres. to staples)
        # plaq, means effective open plaquettes, which are SU(n) matrices.
        plaq_0, staples_sv = self.staples_handle.staple(x_mu, staples=staples)

        plaq_1, logJ = super().forward(
                plaq_0, staples_sv=staples_sv, log0=log0, reduce_=True
                )
        plaq_0 = None  # because reduce_ is set to True above

        x_mu = self.staples_handle.push2links(
                x_mu, eff_proj_plaq_old=plaq_0, eff_proj_plaq_new=plaq_1
                )

        x[:, self.mu] = x_mu

        return x, logJ

    def backward(self, x, log0=0):
        x_mu = x[:, self.mu]

        staples = self.staples_handle.calc_staples(
                x, mu=self.mu, nu_list=self.nu_list
                )
        # below, sv stands for singular values (corres. to staples)
        plaq_0, staples_sv = self.staples_handle.staple(x_mu, staples=staples)

        plaq_1, logJ = super().backward(
                plaq_0, staples_sv=staples_sv, log0=log0, reduce_=True
                )
        plaq_0 = None  # because reduce_ is set to True above

        x_mu = self.staples_handle.push2links(
                x_mu, eff_proj_plaq_old=plaq_0, eff_proj_plaq_new=plaq_1
                )

        x[:, self.mu] = x_mu

        return x, logJ

    def _hack(self, x, log0=0):
        """Similar to the forward method, but returns intermediate parts."""
        x_mu = x[:, self.mu]

        staples = self.staples_handle.calc_staples(
                x, mu=self.mu, nu_list=self.nu_list
                )
        stack = [(x_mu, staples)]
        # below, sv stands for singular values (corres. to staples)
        plaq_0, staples_sv = self.staples_handle.staple(x_mu, staples=staples)
        stack.append((plaq_0, staples_sv))

        plaq_1, logJ = super().forward(
                plaq_0, staples_sv=staples_sv, log0=log0, reduce_=True
                )
        plaq_0 = None  # because reduce_ is set to True above
        stack.append(super()._hack(plaq_0, staples_sv=staples_sv))
        stack.append((plaq_1, staples_sv, logJ))

        x_mu = self.staples_handle.push2links(
                x_mu, eff_proj_plaq_old=plaq_0, eff_proj_plaq_new=plaq_1
                )
        stack.append((x_mu,))

        x[:, self.mu] = x_mu

        return stack

    def transfer(self, **kwargs):
        return self.__class__(
                self.net_.transfer(**kwargs),
                mu=self.mu,
                nu_list=self.nu_list,
                mask=self.mask,
                staples_handle=self.staples_handle,
                matrix_handle=self.matrix_handle,
                label=self.label
                )
