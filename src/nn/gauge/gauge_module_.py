# Copyright (c) 2021-2022 Javad Komijani

"""This module contains new neural networks for transforming planar gauge
fields.

The classes defined here are children of MatrixModule_ (and in turn Module_),
and the trailing underscore implies that the associated forward and backward
methods handle the Jacobians of the transformation.
"""


import torch

from ..matrix.matrix_module_ import MatrixModule_


def ddp_wrapper(func):
    def identity(x):
        return x
    # This resolves a problem of in-place modified tensors in .forward() call
    def wrapper(*args, **kwargs):
        with torch.autograd.graph.saved_tensors_hooks(pack_hook=identity, unpack_hook=identity):
            output = func(*args, **kwargs)
        return output
    return wrapper


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
    
    mask : instance of Mask
        used for partitioning the links and their corresponding staples to
        active and frozen. Note that mask should be instantiated with e.g.
        split_form='directional_even-odd'.

    staple_handle: class instance
         A class instance to handle matrices as expected in `self._kernel`

    matrix_handle: class instance
         A class instance to handle matrices as expected in `self._kernel`
    """

    def __init__(
            self, net_, *, mu, nu_list, staples_handle, matrix_handle, mask, label="gauge_"
            ):
        super().__init__(net_, matrix_handle=matrix_handle, label=label)
        self.mu = mu
        self.nu_list = nu_list
        self.mask = mask
        self.staples_handle = staples_handle

    @ddp_wrapper
    def forward(self, x, log0=0):
        staples = \
            self.staples_handle.calc_staples(x, mu=self.mu, nu_list=self.nu_list)

        x_mu = x[:, self.mu]
        x_mu, _ = self.staples_handle.staple(x_mu, staples=staples)
        x_mu, logJ = super().forward(x_mu, log0)
        x_mu = self.staples_handle.unstaple(x_mu)

        x[:, self.mu] = x_mu

        return x, logJ

    @ddp_wrapper
    def backward(self, x, log0=0):
        staples = \
            self.staples_handle.calc_staples(x, mu=self.mu, nu_list=self.nu_list)

        x_mu = x[:, self.mu]
        x_mu, _ = self.staples_handle.staple(x_mu, staples=staples)
        x_mu, logJ = super().backward(x_mu, log0)
        x_mu = self.staples_handle.unstaple(x_mu)

        x[:, self.mu] = x_mu

        return x, logJ

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


# =============================================================================
class SVDMatrixModule_:
    pass


class SVDGaugeModule_(SVDMatrixModule_):
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
    
    mask : instance of Mask
        used for partitioning the links and their corresponding staples to
        active and frozen. Note that mask should be instantiated with e.g.
        split_form='directional_even-odd'.

    staple_handle: class instance
         A class instance to handle matrices as expected in `self._kernel`

    matrix_handle: class instance
         A class instance to handle matrices as expected in `self._kernel`
    """

    def __init__(
            self, net_, *, mu, nu_list, staples_handle, matrix_handle, mask, label="gauge_"
            ):
        super().__init__(net_, matrix_handle=matrix_handle, label=label)
        self.mu = mu
        self.nu_list = nu_list
        self.mask = mask
        self.staples_handle = staples_handle

    @ddp_wrapper
    def forward(self, x, log0=0):
        x_mu = x[:, self.mu]
        x_mu_active, x_mu_frozen = self.mask.split(x_mu)

        staples_frozen, _ = self.mask.split(
            self.staples_handle.calc_staples(x, mu=self.mu, nu_list=self.nu_list)
            )

        x_mu_active, svd_frozen = \
              self.staples_handle.staple(x_mu_active, staples=staples_frozen)

        x_mu_active, logJ = super().forward([x_mu_active, svd_frozen], log0)
        x_mu_active = self.staples_handle.unstaple(x_mu_active)

        x_mu = self.mask.cat(x, x_mu_frozen)
        x[:, self.mu] = x_mu

        return x, logJ

    @ddp_wrapper
    def backward(self, x, log0=0):
        x_mu = x[:, self.mu]
        x_mu_active, x_mu_frozen = self.mask.split(x_mu)

        staples_frozen, _ = self.mask.split(
            self.staples_handle.calc_staples(x, mu=self.mu, nu_list=self.nu_list)
            )

        x_mu_active, svd_frozen = \
              self.staples_handle.staple(x_mu_active, staples=staples_frozen)

        x_mu_active, logJ = super().backward([x_mu_active, svd_frozen], log0)
        x_mu_active = self.staples_handle.unstaple(x_mu_active)

        x_mu = self.mask.cat(x, x_mu_frozen)
        x[:, self.mu] = x_mu

        return x, logJ

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
