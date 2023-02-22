# Copyright (c) 2023 Javad Komijani

"""This is a module for including the determinant of fermion propagators."""


import torch

from fermionic_tools.staggered import dirac_dagger_dirac_operator


class LogDetAction:
    """
    The effective action corrresponding to deteriminat of fermion propagators
    as:

    .. math::
        \sqrt{|M|} = e^{\log |M|} = e^{- S_{eff}}
    """

    def __init__(self, fermions_dict={0: dict(mass=1, copies=1)}):
        self.fermions_dict = fermions_dict

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
        action = -self.calc_logdet(cfgs)
        if subtractive_term is not None:
            action -= subtractive_term
        return action

    def calc_logdet(self, cfgs):
        fermions = self.fermions_dict
        logdet = 0
        for fermion in self.fermions_dict.values():
            ddd_ee = dirac_dagger_dirac_operator(
                    links=cfgs, lat_shape=cfgs.shape[2:], mass=fermion['mass'],
                    link_shape=[], vector_axis=1, eo_partition='ee'
                )
            # Because we calculate determinant of dirac_dagger_dirac, logdet
            # must be divided by two, but since we only take derivative of the
            # even-even partition, logdet should be multiplied by 2; hence, we
            # the total over coefficient is 1 for each copy of the fermion with
            # given mass.
            det = torch.linalg.det(ddd_ee.to_dense()).real
            logdet += torch.log(det) * fermion['copies']
        return logdet

    def log_prob(self, x, action_logz=0):
        """Returns log probability up to an additive constant."""
        return -self.action(x) - action_logz

