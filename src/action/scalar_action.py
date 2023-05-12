# Copyright (c) 2021-2022 Javad Komijani

"""This is a module for defining actions..."""


import torch


class ScalarPhi4Action:
    r"""The action is defined as

    .. math::

        S = \int d^n x (
            \frac{\kappa}{2} (\partial_\mu \phi(x))^2
            + \frac{m^2}{2} \phi(x)^2
            + \lambda \phi(x)^4
            ).
    """
    def __init__(self,
            *, m_sq, lambd, ndim, kappa=1, a=1, propagate_density=False
            ):
        # We first absorb the lattice spacing in the parameters of the action
        kappa, m_sq, lambd = kappa * a**(ndim-2), m_sq * a**ndim, lambd * a**ndim
        self.w_0 = 0.5 * (2 * kappa)
        self.w_2 = 0.5 * (m_sq + 2 * kappa * ndim)
        self.w_4 = lambd
        self.ndim = ndim
        self._propagate_density = propagate_density  # for test
        self.parameters = dict(m_sq=m_sq, lambd=lambd, ndim=ndim, kappa=kappa, a=a)

    def _set_propagate_density(self, propagate_density):  # for test
        self._propagate_density = propagate_density

    def reset_parameters(self, *, m_sq, lambd, kappa=1, a=1):
        # We first absorb the lattice spacing in the parameters of the action
        ndim = self.ndim
        kappa, m_sq, lambd = kappa * a**(ndim-2), m_sq * a**ndim, lambd * a**ndim
        self.w_0 = 0.5 * (2 * kappa)
        self.w_2 = 0.5 * (m_sq + 2 * kappa * ndim)
        self.w_4 = lambd
        self.parameters['m_sq'] = m_sq
        self.parameters['lambd'] = lambd
        self.parameters['kappa'] = kappa
        self.parameters['a'] = a
       
    def __call__(self, cfgs, subtractive_term=None):
        if self._propagate_density:
            return self.action_density(cfgs, subtractive_term)
        else:
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
        dim = tuple(range(-self.ndim, 0, 1))  # 0 axis might refer to batches
        if self.w_0 == 0:
            w2, w4 = self.w_2, self.w_4
            action = torch.sum(w2 * cfgs**2 + w4 * cfgs**4, dim=dim)
        else:
            w2, w4 = self.w_2/self.w_0, self.w_4/self.w_0
            action = torch.sum(w2 * cfgs**2 + w4 * cfgs**4, dim=dim)
            for mu in dim:
                action -= torch.sum(cfgs * torch.roll(cfgs, 1, mu), dim=dim)
            action *= self.w_0

        if subtractive_term is not None:
            action -= subtractive_term
        return action

    def action_density(self, cfgs, subtractive_term=None):
        """
        Parameters
        ----------
        cfgs : tensor
            Tensor of configurations
        subtractive_term: None/scalar/tensor (optional)
            If not None, this term gets subtracted from action
        """
        dim = tuple(range(-self.ndim, 0, 1))  # 0 axis might refer to batches
        if self.w_0 == 0:
            w2, w4 = self.w_2, self.w_4
            action_density = w2 * cfgs**2 + w4 * cfgs**4
        else:
            # Action density is not unique; let us use a version of it that
            # is symmetric and also its kinetic term is always positive.
            w_0 = self.w_0 / 4
            w2, w4 = (self.w_2 - self.w_0 * self.ndim)/w_0, self.w_4/w_0
            action_density = w2 * cfgs**2 + w4 * cfgs**4
            roll = torch.roll
            for mu in dim:
                # action_density -= cfgs * (roll(cfgs, 1, mu) + roll(cfgs, -1, mu))
                action_density += (cfgs - roll(cfgs, -1, mu))**2
                action_density += (cfgs - roll(cfgs, +1, mu))**2
            action_density *= w_0

        if subtractive_term is not None:
            action_density -= subtractive_term
        return action_density

    def potential(self, x):
        m_sq = self.w_2 - self.ndim * self.w_0
        lambd = self.w_4
        return m_sq * x**2 + lambd * x**4

    def log_prob(self, x, action_logz=0):
        """Returns log probability up to an additive constant."""
        return -self.action(x) - action_logz
