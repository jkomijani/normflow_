# Copyright (c) 2021-2022 Javad Komijani

"""This module introduces a neural network to handle the PSD of a field."""


from .._normflowcore import np, torch, Module_


class PSDBlock_(Module_):
    """Power Spectral Density Block"""

    def __init__(self, *, mfnet_, fftnet_, label='psd-block'):
        super().__init__(label=label)
        self.mfnet_ = mfnet_
        self.fftnet_ = fftnet_

    def forward(self, x, log0=0):
        dim = list(range(1, x.dim()))
        rvol = np.product(x.shape[1:])**0.5  # square root of volume
        x_mean = torch.mean(x, dim=dim).reshape(-1, *[1 for _ in dim])
        y_mf, logJ_mf = self.mfnet_.forward(x_mean, rvol=rvol)
        y_fft, logJ_fft = self.fftnet_.forward(x - x_mean)
        return (y_mf + y_fft), (log0 + logJ_mf + logJ_fft)

    def backward(self, x, log0=0):
        dim = list(range(1, x.dim()))
        rvol = np.product(x.shape[1:])**0.5  # square root of volume
        x_mean = torch.mean(x, dim=dim).reshape(-1, *[1 for _ in dim])
        y_mf, logJ_mf = self.mfnet_.backward(x_mean, rvol=rvol)
        y_fft, logJ_fft = self.fftnet_.backward(x - x_mean)
        return (y_mf + y_fft), (log0 + logJ_mf + logJ_fft)

    def hack(self, x, log0=0):
        """Similar to the forward method, except gives 2 parts seperately..."""
        dim = list(range(1, x.dim()))
        rvol = np.product(x.shape[1:])**0.5  # square root of volume
        x_mean = torch.mean(x, dim=dim).reshape(-1, *[1 for _ in dim])
        y_mf, logJ_mf = self.mfnet_.forward(x_mean, rvol=rvol)
        y_fft, logJ_fft = self.fftnet_.forward(x - x_mean)
        stack = [(x_mean, log0), (y_mf, logJ_mf), (y_fft, logJ_fft),
                 ((y_mf + y_fft), (log0 + logJ_mf + logJ_fft))
                ]
        return stack

    def transfer(self, **kwargs):
        return PSDBlock_(
                mfnet_ = self.mfnet_.transfer(**kwargs),
                fftnet_ = self.fftnet_.transfer(**kwargs)
                )
