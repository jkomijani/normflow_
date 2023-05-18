# Copyright (c) 2023 Javad Komijani

"""This is a module for defining actions related to Ginibre Ensemble."""

import torch
import numpy as np

from ..lib.linalg import haar_sqr  # special qr (sqr) decomposition


class GinibreGaugeAction:
    r"""An action with two pieces for seperate handling of the Q and R
    matrices of QR-decomposed GL(n, C) matrices.

    After QR decomposition of GL(n, C) matrices, the Q part is weighted
    according to GaugeAction or MatrixAction, while R is weighted according to
    the Ginibre action, which treates the components as iid normal complex
    variables. By the way, note that wetghting R with the Ginibre action is
    equal to weighting the orginal GL(n, C) matrix.

    The QR decomposition is implemented following `[Mezzadri]`_, which takes
    advantage of the Ginibre random matrices to generate unitary matrices.

    .. _[Mezzadri]:
        F. Mezzadri,
        "How to generate random matrices from the classical compact groups",
        :arXiv:`math-ph/0609050`.

    Parameters
    ----------
    gauge_action : Action
        for weighting the SUn(n) part of the Q part of QR decomposition.
    ginibre_prior : Prior
        for weighting the R part of QR decomposition.
    """
    def __init__(self, *, gauge_action, ginibre_prior):
        self.gauge_action = gauge_action
        self.ginibre_prior = ginibre_prior  # only log_prob method will be used

    def __call__(self, cfgs):
        return self.action(cfgs)

    def action(self, cfgs):
        """Return the action value of the input cofigurations."""
        q_sun, q_u1, r = self.decompose(cfgs)  # SU(n) x U(1) x Triangular
        action = self.gauge_action(q_sun) - self.ginibre_prior.log_prob(cfgs)
        # Note:
        # 1. self.ginibre_prior.log_prob(r) = self.ginibre_prior.log_prob(cfgs)
        # 2. log_prob(q_u1) is assumed to be 0.
        return action
    
    def action_density(self, cfgs):
        """Return the action density of the input cofigurations."""
        q_sun, q_u1, r = self.decompose(cfgs)  # SU(n) x U(1) x Triangular
        action_density = self.gauge_action.action_density(q_sun) \
                        - self.ginibre_prior.dist.log_prob(cfgs)
        # Note:
        # 1. self.ginibre_prior.log_prob(r) = self.ginibre_prior.log_prob(cfgs)
        # 2. log_prob(q_u1) is assumed to be 0.
        return action_density
    
    @staticmethod
    def decompose(x, u1_matrix_rep=False):
        """Decompose x to SU(n), U(1) & invertible upper triangular matrix."""
        return haar_sqr(x, u1_matrix_rep=u1_matrix_rep)
