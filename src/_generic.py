# Copyright (c) 2021-2022 Javad Komijani

import torch
import numpy as np


# =============================================================================
class Resampler:
    """
    Parameters:
    -----------
    method : str (option)
       The default method of resampling is bootstrap, the other option is
       jackknife, which can be invoked with option set to 'jackknife'.
    """
    def __init__(self, method='bootstrap'):
        self.method = method

    def __call__(self, samples, n_resamples=100, binsize=1, batch_size=None):
        """
        Parameters:
        -----------
        samples: tensor/ndarray

        n_resamples: int (option)
            Irrelavant for the jackknife method.

        binsize: int (option)
            Bins the data before sampling from it.
        """
        l_b = samples.shape[0] // binsize  # lenght of binned samples
        resample_shape = (l_b * binsize, *samples.shape[1:])
        binned_samples = samples[:(l_b * binsize)].reshape(l_b, binsize, -1)

        if batch_size is None:
            batch_size = l_b  # useful if method is not 'jackknife'

        if type(samples) == torch.Tensor:
            arange, randint = torch.arange, torch.randint
        else:
            arange, randint = np.arange, np.random.randint

        if self.method == 'jackknife':
            n_resamples = l_b
            get_indices = lambda i: arange(l_b)[arange(l_b) != i]
        else:
            get_indices = lambda i: randint(l_b, size=(batch_size,))

        for i in range(n_resamples):
            yield binned_samples[get_indices(i)].reshape(*resample_shape)


# =============================================================================
def grab(var):
    # `var.detach()` returns a new Tensor, detached from the current graph and
    # never require gradient. `var.cpu()` copies `var` from GPU to CPU.
    return var.detach().cpu().numpy()


def estimate_logz(logqp, n_resamples=20):
    """Estimate log(z) from logqp = log(q) - log(p * z) by evaluating

    Integral p * z = Integral q exp(-logqp)

    which is expected to be equal to z when correctly sampled.
    """
    def calc_logz(x):
        return torch.logsumexp(x, dim=0).item() - np.log(logqp.shape[0])
    mean = calc_logz(-logqp)
    std = np.std([calc_logz(x) for x in Resampler()(-logqp, n_resamples)])
    return mean, std


def fmt_value_err(value, error, err_digits=1):
    try:
        digits = -int(np.floor(np.log10(error))) + err_digits - 1
        if digits < 0:
            digits = 0
        str_ = "{0:.{2}f}({1:.0f})".format(value, error * 10**digits, digits)
    except ValueError:
        str_ = "{0}+-{1}".format(value, error)
    return str_
