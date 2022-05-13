# Copyright (c) 2021-2022 Javad Komijani

import torch
import numpy as np

from ._generic import grab, estimate_logz, fmt_value_err
from ._generic import Resampler


# =============================================================================
class MCMCSamplerCore:
    """Perform Monte Carlo Markov chain simulation..."""

    def __init__(self, model):
        self._model = model
        self.history = MCMCHistory()
        self.logqp_ref = None

    @torch.no_grad()
    def serial_sample_generator(self, n_samples, batch_size=16):
        """Generate Monte Carlo Markov Chain samples one by one"""
        for i in range(n_samples):
            ind = i % batch_size  # the index of the batch
            if ind == 0:
                y, logqp = self.sample_(batch_size)  # logqp is logq - logp
            yield y[ind].unsqueeze(0), logqp[ind].unsqueeze(0)

    @torch.no_grad()
    def calc_accept_rate(self, n_samples=1024, n_resamples=20, asstr=False):
        """Calculate acceptance rate from logqp = log(q) - log(p * z)"""
        # First, draw (raw) samples
        logqp = grab(self._model.raw_dist.sample_(n_samples)[1])
        # Now calculate the mean and std (by bootstraping) of acceptance rate
        def calc_rate(logqp):
            return np.mean(Metropolis.calc_accept_status(logqp))
        mean = calc_rate(logqp)
        std = np.std([calc_rate(x) for x in Resampler()(logqp, n_resamples)])
        if asstr:
            return fmt_value_err(mean, std, err_digits=1)
        else:
            return mean, std


# =============================================================================
class BlockedMCMCSampler(MCMCSamplerCore):
    pass


# =============================================================================
class MCMCSampler(MCMCSamplerCore):
    """Perform Monte Carlo Markov chain simulation..."""

    @torch.no_grad()
    def sample_(self, batch_size=1, describe=False):
        """Return a batch of Monte Carlo Markov Chain samples generated using
        independence Metropolis method.
        Acceptances/rejections occur proportionally to how well/poorly
        the model density matches the desired density. Decreasing block_len
        """
        # create empty tensor with the size of interest to save accepted stuff
        acpt_y = torch.empty((batch_size, *self._model.prior.shape))  # accepted y
        acpt_logqp = torch.empty((batch_size,))

        logqp_ref = self.logqp_ref
        raw_bsize = batch_size
        n_acpt = 0  # number of accepted samples
        n_true, n_tried = 0, 0
        rate = []

        def append(y, logqp):
            """Append accepted stuff and return the index of last item,
            but make sure number of accepted samples doesn't exceed batch_size.
            """
            n_acpt_new = n_acpt + logqp.shape[0]
            if n_acpt_new > batch_size:  # drop the extra samples
                n_extra, n_acpt_new = n_acpt_new - batch_size, batch_size
                y, logqp = y[:-n_extra], logqp[:-n_extra]
            acpt_y[n_acpt:n_acpt_new] = y
            acpt_logqp[n_acpt:n_acpt_new] = logqp
            return n_acpt_new

        while n_acpt < batch_size:
            # 1. start with raw drawing samples
            y, logqp = self._model.raw_dist.sample_(raw_bsize)

            # 2. filter the samples using Metropolis algorithm & append
            accept_seq = Metropolis.calc_accept_status(grab(logqp), logqp_ref)
            n_acpt = append(y[accept_seq], logqp[accept_seq])
            n_true += sum(accept_seq)  # = n_acpt if there are no extra samples
            n_tried += raw_bsize

            # 3. estimate number of samples we need to try (times 1.5 as margin)
            n_more = int(1.5 * (batch_size - n_acpt) * n_tried / (n_acpt + 1))
            raw_bsize = min(raw_bsize, int(n_more))

            # 4. update logqp_ref for the next round
            logqp_ref = acpt_logqp[n_acpt - 1].item()

        self.logqp_ref = logqp_ref
        self.history.logqp += logqp.tolist()
        self.history.accept_rate.append(n_true/n_tried)

        if describe:
            print(f"Summary of MCMC sample generation")
            print("\t", self.history.report_summary(since=-batch_size, asstr=True))

        return acpt_y, acpt_logqp


# =============================================================================
class MCMCHistory:

    def __init__(self):
        self._init_history()

    def _init_history(self):
        self.logqp = []
        self.accept_rate = []
        self.accept_seq = []
        self.hits = []
        self.moms = []

    def clear_histroy(self):
        self._init_history()

    def report_summary(self, since=0, asstr=False):

        if asstr:
            fmt = lambda mean, std: fmt_value_err(mean, std, err_digits=2)
        else:
            fmt = lambda mean, std: (mean, std)

        logqp = torch.tensor(self.logqp[since:])  # estimate_logz
        accept_rate = torch.tensor(self.accept_rate[since:])
        mean_std = lambda t: (t.mean().item(), t.std().item())

        report = {'logqp': fmt(*mean_std(logqp)),
                  'logz': fmt(*estimate_logz(logqp)),
                  'accept_rate': fmt(*mean_std(accept_rate))
                  }
        return report


# =============================================================================
class Metropolis:

    @staticmethod
    @torch.no_grad()
    def calc_accept_status(logqp, logqp_ref=None):
        """Returns accept/reject using Metropolis algorithm."""
        # Much faster if inputs are np.ndarray & python number (NO tensor)
        if logqp_ref is None:
            logqp_ref = logqp.mean()
        status = np.empty(len(logqp), dtype=bool)
        rand_arr = np.log(np.random.rand(logqp.shape[0]))
        for i, logqp_i in enumerate(logqp):
            status[i] = rand_arr[i] < (logqp_ref - logqp_i)
            if status[i]:
                logqp_ref = logqp_i
        return status

    @staticmethod
    def calc_accept_count(status):
        """Count how many repetition till next accepted configuration."""
        ind = np.where(status)[0]  # index of True ones
        mul = ind[1:] - ind[:-1]  # count except for the last
        return ind[0], list(mul) + [len(states) - ind[-1]]


# =============================================================================
