# Copyright (c) 2021-2022 Javad Komijani

"""This module contains new neural networks for transforming planar gauge
fields.

The classes defined here are children of MatrixModule_ (and in turn Module_),
and the trailing underscore implies that the associated forward and backward
methods handle the Jacobians of the transformation.
"""


import torch

from ..matrix.matrix_module_ import MatrixModule_


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
        plaq_1, logJ = super().forward(plaq_0, log0)
        x_mu = self.plaq_handle.push_plaq2links(
                new_plaq=plaq_1, old_plaq=plaq_0, links=x_mu, zpmask=self.zpmask
                )
        return x_mu, logJ

    def backward(self, x_mu, x_nu, log0=0):
        plaq_0 = self.plaq_handle.calc_zpmasked_open_plaq(x_mu, x_nu, self.zpmask)
        plaq_1, logJ = super().backward(plaq_0, log0)
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
