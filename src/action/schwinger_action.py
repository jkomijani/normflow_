# Copyright (c) 2023 Javad Komijani

"""This is a module for Schwinger model's actions..."""


import torch

from .gauge_action import U1GaugeAction
from .logdet_action import LogDetAction
from fermionic_tools.staggered import dirac_dagger_dirac_operator


class SchwingerAction(U1GaugeAction):
    r"""The action for Schwinger model:

    .. math::

        S = \int d^n x (...).
    """
    def __init__(self, *, beta, ndim,
            fermions_dict={0: dict(mass=1, copies=1)}
            ):
        super().__init__(beta=beta, ndim=ndim)
        self.fermions_dict = fermions_dict
        self.fermions = LogDetAction(fermions_dict)

    def __call__(self, cfgs, subtractive_term=None):
        return self.action(cfgs, subtractive_term)

    def action(self, cfgs, subtractive_term=None):
        """
        Parameters
        ----------
        cfgs : tensor
            Tensor of configurations
        subtractive_term: None/scalar/tensor (optional)
            If not None, this term gets subtracted from action
        """
        action = super().action(cfgs, subtractive_term=subtractive_term)
        action -= self.fermions.calc_logdet(cfgs)
        return action

    def action_density(self, cfgs, subtractive_term=None):
        pass

    @property
    def parameters(self):
        return dict(beta=self.beta, ndim=self.ndim, fermions=self.fermions.fermions_dict)