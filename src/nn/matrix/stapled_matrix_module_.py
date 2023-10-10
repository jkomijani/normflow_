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
    """A module for transforming matrices.

    Parameters
    ----------
    param_net_: instance of Module_ or ModuleList_
        to change a set of parameters corresponding to the matrices links as
        specfied in the upper class `MatrixModule_`.

    matrix_handle: class instance
        for parametrization of the matrices. For more information on how it is
        used, see `self._kernel`.
    """
    def __init__(self, dual_param_net_, param_net_,
            *, matrix_handle, label="matrix_module_"
            ):
        super().__init__(label=label)
        self.dual_param_net_ = dual_param_net_
        self.param_net_ = param_net_
        self.matrix_handle = matrix_handle

    def forward(self, x, *, svd_, log0=0, reduce_=False):
        return self._kernel(
                x, svd_=svd_, log0=log0, reduce_=reduce_, forward=True
                )

    def backward(self, x, *, svd_, log0=0, reduce_=False):
        return self._kernel(
                x, svd_=svd_, log0=log0, reduce_=reduce_, forward=False
                )

    def _kernel(self, matrix, *, svd_, forward, reduce_, log0=0):
        """Return the transformed matrix and its Jacobian.

        To this end, `matrix_handle` is used for parametrizing the input
        matrix. Then `param_net_` is used to transform the parameters.
        Finally, `matrix_handle` is used to construct a new matrix from the
        transformed parameters.
        """
        # 1. Parametrize the input matrix
        param, logJ_mat2par = self.matrix_handle.matrix2param_(matrix)

        # 2. Move the channel axis, in which the param are listed, from -1 to 1
        param = torch.movedim(param, -1, 1)
        singv = torch.movedim(svd_.S, -1, 1)  # singular values

        # 3. Transform param
        if forward:
            param, logJ_dualpar2par = self.dual_param_net_.forward(param, singv)
            param, logJ_par2par = self.param_net_.forward(param)
        else:
            param, logJ_par2par = self.param_net_.backward(param)
            param, logJ_dualpar2par = self.dual_param_net_.backward(param, singv)

        # 4. Move back the channel axis to -1
        param = torch.movedim(param, 1, -1)  # return channel axis to -1

        # 5. Construct a new matrix from the transformed parameters
        matrix, logJ_par2mat = \
                self.matrix_handle.param2matrix_(param, reduce_=reduce_)

        # 6. Add up all log-Jacobians
        logJ = logJ_mat2par + logJ_dualpar2par + logJ_par2par + logJ_par2mat

        return matrix, log0 + logJ

    def _hack(self, matrix, *, svd_, log0=0, forward=True, reduce_=False):
        """Similar to the forward/backward methods, but returns intermediate
        parts too.
        """
        # 1. Parametrize the input matrix
        param, logJ_mat2par = self.matrix_handle.matrix2param_(matrix)
        stack = [(param, logJ_mat2par)]

        # 2. Move the channel axis, in which the param are listed, from -1 to 1
        param = torch.movedim(param, -1, 1)  # move channel axis from -1 to 1
        singv = torch.movedim(svd_.S, -1, 1)  # singular values

        # 3. Transform param
        if forward:
            param, logJ_dualpar2par = self.dual_param_net_.forward(param, singv)
            stack.append((param, logJ_dualpar2par))
            param, logJ_par2par = self.param_net_.forward(param)
            stack.append((param, logJ_par2par))
        else:
            param, logJ_par2par = self.param_net_.backward(param)
            stack.append((param, logJ_par2par))
            param, logJ_dualpar2par = self.dual_param_net_.backward(param, singv)
            stack.append((param, logJ_dualpar2par))

        # 4. Move back the channel axis to -1
        param = torch.movedim(param, 1, -1)  # return channel axis to -1

        # 5. Construct a new matrix from the transformed parameters
        matrix, logJ_par2mat = \
        matrix, logJ_par2mat = \
                self.matrix_handle.param2matrix_(param, reduce_=reduce_)
        stack.append((matrix, logJ_par2mat))

        # 6. Add up all log-Jacobians
        logJ = logJ_mat2par + logJ_dualpar2par + logJ_par2par + logJ_par2mat
        stack.append((matrix, log0 + logJ))

        return stack

    def transfer(self, **kwargs):
        return self.__class__(self.dual_param_net_.transfer(**kwargs),
                             self.param_net_.transfer(**kwargs),
                             matrix_handle=self.matrix_handle,
                             label=self.label
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
