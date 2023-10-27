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

    vector_axis = 1
    unbind_vector_axis = True  # the vector axis then will be moved to 0

    def forward(self, x, log0=0):
        if self.unbind_vector_axis:
            x = list(torch.unbind(x, self.vector_axis))
            x, log0 = super().forward(x, log0)
            return torch.stack(x, dim=self.vector_axis), log0
        else:
            return super().forward(x.clone(), log0)

    def backward(self, x, log0=0):
        if self.unbind_vector_axis:
            x = list(torch.unbind(x, self.vector_axis))
            x, log0 = super().backward(x, log0)
            return torch.stack(x, dim=self.vector_axis), log0
        else:
            return super().backward(x.clone(), log0)

    def hack(self, x, log0=0):
        """Similar to the forward method, except that returns the output of
        middle blocks too; useful for examining effects of each block.
        """
        stack = [(x, log0)]

        if self.unbind_vector_axis:
            for net_ in self:
                x = list(torch.unbind(x, self.vector_axis))
                x, log0 = net_(x, log0)
                x = torch.stack(x, dim=self.vector_axis)
                stack.append([x, log0])
            return stack
        else:
            return None


# =============================================================================
class GaugeModule_(MatrixModule_):
    """
    Parameters
    ----------
    param_net_: instance of Module_ or ModuleList_
        a core network to change a set of parameters corresponding to the
        stapled links as specified in the supper class `MatrixModule_`.

    mu : int
        specifies the direction of links that are going to be changed
    
    nu_list : list of int
        (in combination w/ mu) specifies the plane of staples to be calculated
    
    staple_handle: class instance
        to calculate staples and use them.

    matrix_handle: class instance
        to handle matrices as expected in the supper class `MatrixModule_`.
    """

    unbounded_vector_axis = True

    def __init__(self, param_net_,
            *, mu, nu_list, staples_handle, matrix_handle, label="gauge_"
            ):
        super().__init__(param_net_, matrix_handle=matrix_handle)
        self.mu = mu
        self.nu_list = nu_list
        self.staples_handle = staples_handle
        self.label = label

    def forward(self, x, log0=0):
        if self.unbounded_vector_axis:
            x_mu = x[self.mu]
        else:
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

        if self.unbounded_vector_axis:
            x[self.mu] = x_mu
        else:
            x[:, self.mu] = x_mu

        return x, logJ

    def backward(self, x, log0=0):
        if self.unbounded_vector_axis:
            x_mu = x[self.mu]
        else:
            x_mu = x[:, self.mu]

        staples = self.staples_handle.calc_staples(
                x, mu=self.mu, nu_list=self.nu_list
                )
        slink, svd_ = self.staples_handle.staple(x_mu, staples=staples)

        slink_rotation, logJ = super().backward(slink, log0, reduce_=True)

        x_mu = self.staples_handle.push2link(
                x_mu, slink_rotation=slink_rotation, svd_=svd_
                )

        if self.unbounded_vector_axis:
            x[self.mu] = x_mu
        else:
            x[:, self.mu] = x_mu

        return x, logJ

    def _hack(self, x, forward=True):
        """Similar to the forward method, but returns intermediate parts."""

        if self.unbounded_vector_axis:
            x_mu = x[self.mu]
        else:
            x_mu = x[:, self.mu]

        staples = self.staples_handle.calc_staples(
                x, mu=self.mu, nu_list=self.nu_list
                )
        slink, svd_ = self.staples_handle.staple(x_mu, staples=staples)

        if forward:
            slink_rotation, logJ = super().forward(slink, reduce_=True)
        else:
            slink_rotation, logJ = super().backward(slink, reduce_=True)

        stack = dict(
                x_mu_initial = x_mu,
                staples = staples,
                slink = slink,
                svd_ = svd_,
                slink_rotation = slink_rotation,
                logJ = logJ,
                super_hack = super()._hack(slink, forward=forward, reduce_=True)
                )

        x_mu = self.staples_handle.push2link(
                x_mu, slink_rotation=slink_rotation, svd_=svd_
                )
        stack["x_mu_final"] = x_mu

        return stack

    def transfer(self, **kwargs):
        return self.__class__(
                self.param_net_.transfer(**kwargs),
                mu=self.mu,
                nu_list=self.nu_list,
                staples_handle=self.staples_handle,
                matrix_handle=self.matrix_handle,
                label=self.label
                )


# =============================================================================
class SVDGaugeModule_(StapledMatrixModule_):
    """
    Similar to GaugeModule_ but uses singular values of the staples for
    processing too.

    Parameters
    ----------
    dual_param_net_: instance of Module_ or ModuleList_
        a core network to change a set of parameters corresponding to the
        stapled links as specified in the supper class `StapledMatrixModule_`.

    param_net_: instance of Module_ or ModuleList_
        a core network to change a set of parameters corresponding to the
        stapled links as specefied in the supper class `StapledMatrixModule_`.

    mu : int
        specifies the direction of links that are going to be changed
    
    nu_list : list of int
        (in combination w/ mu) specifies the plane of staples to be calculated
    
    staple_handle: class instance
        to calculate staples and use them.

    matrix_handle: class instance
        to handle matrices as expected in the supper class `MatrixModule_`.
    """

    unbounded_vector_axis = True

    def __init__(self, dual_param_net_, param_net_,
            *, mu, nu_list, staples_handle, matrix_handle, label="gauge_",
            **kwargs
            ):
        super().__init__(
            dual_param_net_, param_net_, matrix_handle=matrix_handle, **kwargs
            )
        self.mu = mu
        self.nu_list = nu_list
        self.staples_handle = staples_handle
        self.label = label

    def forward(self, x, log0=0):
        if self.unbounded_vector_axis:
            x_mu = x[self.mu]
        else:
            x_mu = x[:, self.mu]

        staples = self.staples_handle.calc_staples(
                x, mu=self.mu, nu_list=self.nu_list
                )
        # slink: stapled link
        slink, svd_ = self.staples_handle.staple(x_mu, staples=staples)

        slink_rotation, logJ = super().forward(
                slink, log0=log0, svd_=svd_, reduce_=True
                )

        x_mu = self.staples_handle.push2link(
                x_mu, slink_rotation=slink_rotation, svd_=svd_
                )

        if self.unbounded_vector_axis:
            x[self.mu] = x_mu
        else:
            x[:, self.mu] = x_mu

        return x, logJ

    def backward(self, x, log0=0):
        if self.unbounded_vector_axis:
            x_mu = x[self.mu]
        else:
            x_mu = x[:, self.mu]

        staples = self.staples_handle.calc_staples(
                x, mu=self.mu, nu_list=self.nu_list
                )
        slink, svd_ = self.staples_handle.staple(x_mu, staples=staples)

        slink_rotation, logJ = super().backward(
                slink, log0=log0, svd_=svd_, reduce_=True
                )

        x_mu = self.staples_handle.push2link(
                x_mu, slink_rotation=slink_rotation, svd_=svd_
                )

        if self.unbounded_vector_axis:
            x[self.mu] = x_mu
        else:
            x[:, self.mu] = x_mu

        return x, logJ

    def _hack(self, x, log0=0, forward=True):
        """Similar to the forward method, but returns intermediate parts."""

        if self.unbounded_vector_axis:
            x_mu = x[self.mu]
        else:
            x_mu = x[:, self.mu]

        staples = self.staples_handle.calc_staples(
                x, mu=self.mu, nu_list=self.nu_list
                )
        stack = [(x_mu, staples)]
        slink, svd_ = self.staples_handle.staple(x_mu, staples=staples)
        stack.append((slink, svd_))

        if forward:
            slink_rotation, logJ = super().forward(
                slink, log0=log0, svd_=svd_, reduce_=True
                )
        else:
            slink_rotation, logJ = super().backward(
                slink, log0=log0, svd_=svd_, reduce_=True
                )

        stack.append(super()._hack(slink, forward=forward, reduce_=True))
        stack.append((slink_rotation,))

        x_mu = self.staples_handle.push2link(
                x_mu, slink_rotation=slink_rotation, svd_=svd_
                )
        stack.append((x_mu,))

        if self.unbounded_vector_axis:
            x[self.mu] = x_mu
        else:
            x[:, self.mu] = x_mu
        stack.append((x, logJ))

        return stack

    def transfer(self, **kwargs):
        return self.__class__(
                self.param_net_.transfer(**kwargs),
                mu=self.mu,
                nu_list=self.nu_list,
                mask=self.mask,
                staples_handle=self.staples_handle,
                matrix_handle=self.matrix_handle,
                label=self.label
                )
