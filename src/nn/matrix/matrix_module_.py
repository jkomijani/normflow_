# Copyright (c) 2021-2022 Javad Komijani

"""This module contains new neural networks for transforming matrices.

The classes defined here are children of Module_, and like Module_, the trailing
underscore implies that the associated forward and backward methods handle the
Jacobians of the transformation.
"""


import torch

from .._core import Module_


# =============================================================================
class MatrixModule_(Module_):

    def __init__(self, net_, *, matrix_handle, label="matrix_module_"):
        """A wrapper to transform matrices using the given network `net_`
        and the parametrization specified in `matrix_handle`.
        For more information on how `matrix_handle` is used, see self._kernel.
        """
        super().__init__(label=label)
        self.net_ = net_
        self.matrix_handle = matrix_handle

    def forward(self, x, log0=0, reduce_=False):
        return self._kernel(x, log0=log0, reduce_=reduce_, forward=True)

    def backward(self, x, log0=0, reduce_=False):
        return self._kernel(x, log0=log0, reduce_=reduce_, forward=False)

    def _kernel(self, matrix, log0=0, forward=True, reduce_=False):
        """Return the transformed matrix and its Jacobian.

        To this end, `matrix_handle` is used for parametrizing the input
        matrix. Then `net_` is used to transform the parameters. Finally,
        `matrix_handle` is used to construct a new matrix from the transformed
        parameters.
        """
        # 1. Parametrize the input matrix
        param, logJ_mat2par = self.matrix_handle.matrix2param_(matrix)

        # 2. Move the channel axis, in which the param are listed, from -1 to 1
        param = torch.movedim(param, -1, 1)

        # 3. Transform param
        if forward:
            param, logJ_par2par = self.net_.forward(param)
        else:
            param, logJ_par2par = self.net_.backward(param)

        # 4. Move back the channel axis to -1
        param = torch.movedim(param, 1, -1)  # return channel axis to -1

        # 5. Construct a new matrix from the transformed parameters
        matrix, logJ_par2mat = \
                self.matrix_handle.param2matrix_(param, reduce_=reduce_)

        # 6. Add up all log-Jacobians
        logJ = logJ_mat2par + logJ_par2par + logJ_par2mat

        return matrix, log0 + logJ

    def _hack(self, matrix, log0=0, reduce_=False):
        """Similar to the forward method, but returns intermediate parts too."""
        param, logJ_mat2par = self.matrix_handle.matrix2param_(matrix)
        stack = [(param, logJ_mat2par)]

        param = torch.movedim(param, -1, 1)  # move channel axis from -1 to 1
        param, logJ_par2par = self.net_(param)
        param = torch.movedim(param, 1, -1)  # return channel axis to -1
        stack.append((param, logJ_par2par))

        matrix, logJ_par2mat = \
                self.matrix_handle.param2matrix_(param, reduce_=reduce_)
        stack.append((matrix, logJ_par2mat))
        return stack

    def transfer(self, **kwargs):
        return self.__class__(self.net_.transfer(**kwargs),
                              matrix_handle=self.matrix_handle, label=self.label
                             )
