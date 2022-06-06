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

    def __init__(
            self, net_, *, zpmask, plaq_handle, matrix_handle, label="pgm_"
            ):
        """
        Parameters
        ----------
        net_: ...

        matrix_handle: class instance
            A class instance to handle matrices as expected in `self._kernel`
        zpmask: stands for zebra planar mask
        """
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

    def transfer(self, **kwargs):
        return self.__class__(
                self.net_.transfer(**kwargs),
                zpmask=self.zpmask,
                plaq_handle=self.plaq_handle,
                matrix_handle=self.matrix_handle,
                label=self.label
                )


# =============================================================================
class NewPlanarGaugeModule_(MatrixModule_):

    def __init__(
            self, net_, *, zpmask, plaq_handle, matrix_handle, label="pgm_"
            ):
        """
        Parameters
        ----------
        net_: ...

        matrix_handle: class instance
            A class instance to handle matrices as expected in `self._kernel`
        zpmask: stands for zebra planar mask
        """
        super().__init__(net_, matrix_handle=matrix_handle, label=label)
        self.zpmask = zpmask
        self.plaq_handle = plaq_handle

    def forward(self, x_mu, x_nu, log0=0):
        func = self.plaq_handle.calc_zpmasked_open_plaqlongplaq
        plaqlongplaq_0 = func(x_mu, x_nu, self.zpmask)
        plaqlongplaq_1, logJ = super().forward(plaqlongplaq_0, log0)
        x_mu = self.plaq_handle.push_plaq2links(
                new_plaq=plaqlongplaq_1[:, 0],
                old_plaq=plaqlongplaq_0[:, 0],
                links=x_mu,
                zpmask=self.zpmask
                )
        return x_mu, logJ

    def backward(self, x_mu, x_nu, log0=0):
        func = self.plaq_handle.calc_zpmasked_open_plaqlongplaq
        plaqlongplaq_0 = func(x_mu, x_nu, self.zpmask)
        plaqlongplaq_1, logJ = super().backward(plaqlongplaq_0, log0)
        x_mu = self.plaq_handle.push_plaq2links(
                new_plaq=plaqlongplaq_1[:, 0],
                old_plaq=plaqlongplaq_0[:, 0],
                links=x_mu,
                zpmask=self.zpmask
                )
        return x_mu, logJ

    def transfer(self, **kwargs):
        return self.__class__(
                self.net_.transfer(**kwargs),
                zpmask=self.zpmask,
                plaq_handle=self.plaq_handle,
                matrix_handle=self.matrix_handle,
                label=self.label
                )
