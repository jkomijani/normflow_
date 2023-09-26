# Copyright (c) 2021-2022 Javad Komijani

"""This module contains new neural networks for transforming matrices.

The classes defined here are children of Module_, and like Module_, the trailing
underscore implies that the associated forward and backward methods handle the
Jacobians of the transformation.
"""


import torch

from .._core import Module_
from .matrix_module_ import MatrixModule_


# =============================================================================
class StapledMatrixModule_(Module_):

    def __init__(self, net_, *, matrix_handle, mask, label="matrix_module_"):
        """A wrapper to transform matrices using the given network `net_`
        and the parametrization specified in `matrix_handle`.
        For more information on how `matrix_handle` is used, see self._kernel.
        """
        super().__init__(label=label)
        self.net_ = net_
        self.mask = mask
        self.matrix_handle = matrix_handle

    def forward(self, x, *, staples_sv, log0=0, reduce_=False):
        return self._kernel(
                self.net_.forward,
                x=x, staples_sv=staples_sv, log0=log0, reduce_=reduce_
                )

    def backward(self, x, *, staples_sv, log0=0, reduce_=False):
        return self._kernel(
                self.net_.backward,
                x=x, staples_sv=staples_sv, log0=log0, reduce_=reduce_
                )

    def _kernel(self, net_method_, *, x, staples_sv, log0=0, reduce_=False):
        """Return the transformed x and its Jacobian.

        Here are the steps:
        1. compute independent parameters corresp. to eigenvalues of matrix x,
        2. transform the parameters,
        3. construct new matrices with the transformed parameters.

        Parameters
        ----------
        staples_sv : tensor
            The singular values (sv) of the staples
        """
        # First, parametrize the (eigenvalues of) matrix x
        param, logJ_mat2par = self.matrix_handle.matrix2param_(x)
        # The channel axis, where the parameterz are listed, should be moved
        # from -1 to 1
        staples_sv = torch.movedim(staples_sv, -1, 1)
        param = torch.movedim(param, -1, 1)

        # Now, mask both eigenvalues and staples singular-values
        param_active, param_frozen = self.mask.split(param)
        sv_frozen, _ = self.mask.split(staples_sv)
        # the staples corresponding to param_active do not vary when
        # param_active vary; that's why they are tagged as frozen.

        out, logJ_par2par = net_method_([param_active, sv_frozen])
        param_active = out[0]  # sv_frozen = out[1]

        param = self.mask.cat(param_active, param_frozen)
        param = torch.movedim(param, 1, -1)  # channel axis to -1

        x, logJ_par2mat = self.matrix_handle.param2matrix_(param, reduce_=reduce_)

        logJ = logJ_mat2par + logJ_par2par + logJ_par2mat

        return x, log0 + logJ

    def _hack(self, x, *, staples_sv, log0=0, reduce_=False):
        """Similar to the forward method, but returns intermediate parts too."""
        param, logJ_mat2par = self.matrix_handle.matrix2param_(x)
        stack = [(param, logJ_mat2par)]

        param_active, param_frozen = self.mask.split(param)
        sv_frozen, _ = self.mask.split(staples_sv)
        sv_frozen = torch.movedim(sv_frozen, -1, 1)
        param_active = torch.movedim(param_active, -1, 1)
        out, logJ_par2par = self.net_([param_active, sv_frozen])
        param_active = out[0]  # sv_frozen = out[1]
        param_active = torch.movedim(param_active, 1, -1)  # channel axis to -1
        param = self.mask.cat(param_active, param_frozen)
        stack.append((param, logJ_par2par))

        x, logJ_par2mat = self.matrix_handle.param2matrix_(param, reduce_=reduce_)
        stack.append((x, logJ_par2mat))

        return stack

    def transfer(self, **kwargs):
        return self.__class__(self.net_.transfer(**kwargs),
                             matrix_handle=self.matrix_handle, label=self.label
                             )


# =============================================================================
class FixedStapledMatrixModule_(MatrixModule_):
    """Similar to  MatrixModel_ except that accepts staples_handle as an option.
    This should be used only for matrix models with fixed staples.
    For guage theories, the staple_handle should not be passed to the matrix
    model because staples keep changing.
    """

    def __init__(self, net_, *, matrix_handle, staples_handle):
        super().__init__(net_, matrix_handle=matrix_handle)
        self.staples_handle = staples_handle

    def forward(self, x, **kwargs):
        x, _ = self.staples_handle.staple(x)
        x, logJ = super().forward(x, **kwargs)
        x = self.staples_handle.unstaple(x)
        return x, logJ

    def backward(self, x, **kwargs):
        x, _ = self.staples_handle.staple(x)
        x, logJ = super().backward(x, **kwargs)
        x = self.staples_handle.unstaple(x)
        return x, logJ

    def _hack(self, x, **kwargs):
        x = self.staples_handle.staple(x)
        stack = [(x, 0)] + super()._hack(x, **kwargs)
        x, logJ = stack[-1]
        x = self.staples_handle.unstaple(x)
        return stack + [(x, 0)]
