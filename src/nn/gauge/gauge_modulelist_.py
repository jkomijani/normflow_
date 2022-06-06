# Copyright (c) 2021-2022 Javad Komijani

"""This module contains new classes for coupling gauge fields."""


import torch
import numpy as np

from .._core import Module_, ModuleList_


# =============================================================================
class GaugeModuleList_(torch.nn.ModuleList):

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

    @property
    def npar(self):
        count = lambda x: np.product(x)
        return sum([count(p.shape) for p in super().parameters()])

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

    def transfer(self, **kwargs):
        return self.__class__([net_.transfer(**kwargs) for net_ in self])
