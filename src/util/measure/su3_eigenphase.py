# Copyright (c) 2021-2022 Javad Komijani

import torch
import numpy as np
import normflow

calc_conjugacy_vol = normflow.lib.eig_handle.SU3MatrixParametrizer.calc_conjugacy_vol

pi = np.pi


class SU3PhaseParametrizer:

    @staticmethod
    def phase2param(phase, gamma=True):  # param: (theta, phi)
        theta = torch.sum(phase**2, dim=-1)**0.5 / 2**0.5
        _, indices = torch.min(torch.abs(phase), dim=-1, keepdim=True)
        min_ = phase.gather(-1, indices).squeeze(-1)
        theta_prime = torch.ones_like(theta)
        theta_prime[theta != 0] = theta[theta != 0]
        phi = torch.asin(min_ /  (theta_prime * 2 / 3**0.5))
        if gamma:
            return theta * torch.cos(phi), phi
        else:
            return theta, phi
    
    @staticmethod
    def param2phase(theta, phi):  # param: (theta, phi)
        shift = 2*pi/3 * torch.tensor([-1, 0, 1]).repeat(*phi.shape, 1)
        return theta.unsqueeze(-1) * 2/3**0.5 * torch.sin(phi.unsqueeze(-1) + shift)
    
    @staticmethod
    def max_principal_theta(phi):
        return pi / torch.cos(phi)
        # torch.cos(phi) > 2 / 3**0.5 torch.sin(phi + 2*pi/3) if  0 < phi  < pi/6
    
    @staticmethod
    def cross_principal_theta(phi):
        return pi * 3**0.5 / 2 / torch.sin(- torch.abs(phi) + 2*pi/3)
    
    @classmethod
    def move_cell(cls, theta, phi, n0=0, n1=0, n2=0):
        assert n0 + n1 + n2 ==0, "the sum of shifts must be zero"
        shift = 2 * pi * torch.tensor([n0, n1, n2])
        new_phase = cls.param2phase(theta, phi) + shift.reshape(*[1]*(phi.ndim - 1), 3)
        return cls.phase2param(new_phase)

    
class SU3ParamMeshGrid:

    def make_data(bins, xlim=[-pi*0.9999/6, pi*0.9999/6], ylim=[0.0001, pi*0.9999]):
        phi = torch.linspace(*xlim, bins)
        gamma = torch.linspace(*ylim, bins)  # gamma = theta * cos(phi)
        Gamma, Phi = torch.meshgrid(gamma, phi, indexing='xy')
        return Gamma, Phi


class SU3PhaseMeshGrid:

    def make_data(bins, xlim=[-pi*0.9999, pi*0.9999], ylim=[-pi, pi]):
        x = torch.linspace(*xlim, bins)
        y = torch.linspace(*ylim, bins)
        X, Y = torch.meshgrid(x, y, indexing='xy')
        Z = -(X + Y)
        return X, Y, Z

    @staticmethod
    def calc_conjugacy_vol(X, Y, Z):
        phase = torch.stack([X, Y, Z], dim=-1)
        V = calc_conjugacy_vol(phase)
        return V.squeeze(-1)  # squeeze the dimension corresp. to phases

    @staticmethod
    def calc_action(X, Y, Z, *, action):
        phase = torch.stack([X, Y, Z], dim=-1)
        M = torch.diag_embed(torch.exp(1j * phase))
        return action.action_density(M)

    def pdf_r(X, Y, Z):
        phase = torch.stack([X, Y, Z], dim=-1)
        V = calc_conjugacy_vol(phase).squeeze(-1)  # squeeze dim of phases
        return V

    def pdf_p(X, Y, Z, *, action):
        phase = torch.stack([X, Y, Z], dim=-1)
        V = calc_conjugacy_vol(phase).squeeze(-1)  # squeeze dim of phases
        S = action.action_density(torch.diag_embed(torch.exp(1j * phase)))
        return V * torch.exp(-S)

    @torch.no_grad()
    def pdf_q(X, Y, Z, *, model):
        # We need to:
        #     1. transform back to (X, Y, Z) to (X', Y', Z')
        #     2. calculate the jacobian of transformation
        #     3. calculate pdf_r(X', Y', Z'), which is the conjugacy volume
        # Below we do not calculate pdf_r(X', Y', Z') because the jocobian of
        # transformation already includes the the conjugacy of volume of both
        # (X, Y, Z) and (X', Y', Z'), but we need to remove the conjugacy
        # volume of (X, Y, Z).
        phase = torch.stack([X, Y, Z], dim=-1)
        y = torch.diag_embed(torch.exp(1j * phase)).reshape(-1, 3, 3)
        _, mlogJ = model.net_.backward(y)  # mlogJ: minus logJ
        V = calc_conjugacy_vol(phase).squeeze(-1)  # squeeze dim of phases
        # mlogJ: already includes the conjugacy volume of input & output
        # V: the conjugacy volume of transforming phase to y, which cancels the
        # corresponding term in mlogJ -> p
        return V * torch.exp(mlogJ.reshape(*X.shape))


def su3_phase_marginal_dist(bins=100, action=None, model=None, to_numpy=True):
    dv = (2 * pi / bins)**2
    X, Y, Z = SU3PhaseMeshGrid.make_data(bins)
    if action is not None:
        P = SU3PhaseMeshGrid.pdf_p(X, Y, Z, action=action)
    elif model is not None:
        P = SU3PhaseMeshGrid.pdf_q(X, Y, Z, model=model)
    else:
        P = SU3PhaseMeshGrid.pdf_r(X, Y, Z, model=model)

    f_marg = torch.sum(P, dim=1)

    func = normflow.seize if to_numpy else (lambda x: x)

    return func(X[0]), func(f_marg / (torch.sum(f_marg) * dv**0.5))
