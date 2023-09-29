# Copyright (c) 2021-2022 Javad Komijani

"""This module contains new neural networks for transforming planar gauge
fields.

The classes defined here are children of MatrixModule_ (and in turn Module_),
and the trailing underscore implies that the associated forward and backward
methods handle the Jacobians of the transformation.
"""


import torch

from .._core import ModuleList_
from ..matrix.matrix_module_ import MatrixModule_


# =============================================================================
class PlanarGaugeModuleList_(ModuleList_):
    """Like ModuleList_, but transforms gauge links according to a pattern as
    described below.


    Parameters
    ----------
    nets_ : a list of instances of Module_ or ModuleList_

    Using net_ to denote each element of nets_, net_ takes as inputs two
    sets of links in directions of mu and nu and transforms the first set in
    the mu direction. The directions mu and nu must be specified in the mask
    used in net_ that we call zpmask, standing for zebra planar mask.
    """

    vector_axis = 1  # note that batch axis is 0

    def __init__(self, nets_, label="gauge_block_"):
        super().__init__(nets_)
        self.label = label

    def forward(self, x, log0=0):
        x = list(torch.unbind(x, self.vector_axis))
        for net_ in self:
            mu = net_.zpmask.mu
            nu = net_.zpmask.nu
            x[mu], log0 = net_.forward(x[mu], x[nu], log0=log0)
        return torch.stack(x, dim=self.vector_axis), log0

    def backward(self, x, log0=0):
        x = list(torch.unbind(x, self.vector_axis))
        for net_ in self[::-1]:
            mu = net_.zpmask.mu
            nu = net_.zpmask.nu
            x[mu], log0 = net_.backward(x[mu], x[nu], log0=log0)
        return torch.stack(x, dim=self.vector_axis), log0

    def hack(self, x, log0=0):
        """Similar to the forward method, except that returns the output of
        middle blocks too; useful for examining effects of each block.
        """
        stack = [(x, log0)]

        x = list(torch.unbind(x, self.vector_axis))
        for net_ in self:
            mu = net_.zpmask.mu
            nu = net_.zpmask.nu
            x[mu], log0 = net_.forward(x[mu], x[nu], log0=log0)
            stack.append((torch.stack(x, dim=self.vector_axis), log0))
        return stack


# =============================================================================
class PlanarGaugeModule_(MatrixModule_):
    """
    Parameters
    ----------
    net_: instance of Module_ or ModuleList_
        net_ takes as inputs two sets of links, which are supposed to be the
        links in directions of mu and nu, and transforms the first set. The
        directions mu and nu are supposed to be the same directions specified
        in the mask given as zpmask.

    matrix_handle: class instance
         A class instance to handle matrices as expected in `self._kernel`

    zpmask: class instance
         zpmask stands for zebra planar mask, which is used to mask the
         links and transform them.
    """

    def __init__(
            self, net_, *, zpmask, plaq_handle, matrix_handle, label="pgm_"
            ):
        super().__init__(net_, matrix_handle=matrix_handle, label=label)
        self.zpmask = zpmask
        self.plaq_handle = plaq_handle

    def forward(self, x_mu, x_nu, log0=0):
        plaq_0 = self.plaq_handle.calc_zpmasked_open_plaq(x_mu, x_nu, self.zpmask)
        plaq_1, logJ = super().forward(plaq_0, log0, reduce_=True)
        plaq_0 = None  # because reduce_ is set to True above
        x_mu = self.plaq_handle.push_plaq2links(
                new_plaq=plaq_1, old_plaq=plaq_0, links=x_mu, zpmask=self.zpmask
                )
        return x_mu, logJ

    def backward(self, x_mu, x_nu, log0=0):
        plaq_0 = self.plaq_handle.calc_zpmasked_open_plaq(x_mu, x_nu, self.zpmask)
        plaq_1, logJ = super().backward(plaq_0, log0, reduce_=True)
        plaq_0 = None  # because reduce_ is set to True above
        x_mu = self.plaq_handle.push_plaq2links(
                new_plaq=plaq_1, old_plaq=plaq_0, links=x_mu, zpmask=self.zpmask
                )
        return x_mu, logJ

    def _hack(self, x_mu, x_nu, log0=0):
        plaq_0 = self.plaq_handle.calc_zpmasked_open_plaq(x_mu, x_nu, self.zpmask)
        return super()._hack(plaq_0, log0)

    def transfer(self, **kwargs):
        return self.__class__(
                self.net_.transfer(**kwargs),
                zpmask=self.zpmask,
                plaq_handle=self.plaq_handle,
                matrix_handle=self.matrix_handle,
                label=self.label
                )
