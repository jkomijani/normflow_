# Copyright (c) 2021-2022 Javad Komijani

"""This is a module containing the core components for normalizing flow.

The central high-level class is called Model, which takes instances of
other classes as input (prior, net_, and action) and provides untilities
to perform training and drawing samples.

Every instance of the central high-level class Model alreay has an instance of
Fitter, which can be used for training.

For drawing samples, one can use ".raw_dist", which does not perform any
Metropolis accept/reject on the samples, or one can use ".mcmc" if Metropolis
accept/reject needed.

Other central classes in this module are Module_, and ModuleList_
that allow us to define neural networks; these two classes are imported and
used by other modules of this package.
"""


import torch
import time

import numpy as np

from .mcmc import MCMCSampler, BlockedMCMCSampler
from .lib.combo import estimate_logz, fmt_val_err
from .device import ModelDeviceHandler


# =============================================================================
class Model:
    """The central high-level class of the package, which
    takes instances of other classes as input (prior, net_, and action)
    and provides untilities to perform training and drawing samples.

    Parameters
    ----------
    prior : An instance of a Prior class (e.g NormalPrior).

    net_ : An instance of ModuleList_ or similar classes. The trailing
        underscore implies that the associate forward method handles
        the Jacobian of the transformation.

    action : An instance of a class that describes the action.

    name : str, option
        A string to label the model
    """

    def __init__(self, *, prior, net_, action, name=None):
        self.name = name
        self.net_ = net_
        self.prior = prior
        self.action = action

        self.fit = Fitter(self)

        self.posterior = Posterior(self)
        self.raw_dist = self.posterior  # alias; todo: remove later
        self.mcmc = MCMCSampler(self)
        self.blocked_mcmc = BlockedMCMCSampler(self)
        self.device_handler = ModelDeviceHandler(self)

    def transform(self, x):
        return self.net_(x)[0]


class Posterior:
    """A class for drawing samples from given model. Note that the samples
    are drawn directly from the model without performing any accept/reject
    filtering.

    Parameters
    ----------
    model : An instance of Model
    """

    def __init__(self, model):
        self._model = model

    @torch.no_grad()
    def sample(self, batch_size=1, **kwargs):
        return self.sample_(batch_size=batch_size, **kwargs)[0]

    @torch.no_grad()
    def sample_(self, batch_size=1, preprocess_func=None):
        """
        Return `batch_size` samples along with `log(q)` and `log(p)`.

        Parameters
        ----------
        batch_size: int
            The size of the samples

        preprocess_func: None or a function
            Introduced to preprocess the prior sample if needed
        """
        x, logr = self._model.prior.sample_(batch_size)
        if preprocess_func is not None:
            x, logr = preprocess_func(x, logr)
        y, logJ = self._model.net_(x)
        logq = logr - logJ
        return y, logq

    @torch.no_grad()
    def sample__(self, batch_size=1, **kwargs):
        y, logq = self.sample_(batch_size=batch_size, **kwargs)
        logp = -self._model.action(y)  # logp is log(p * z)
        return y, logq, logp

    @torch.no_grad()
    def log_prob(self, y):
        """Returns log probability of the samples."""
        x, minus_logJ = self._model.net_.backward(y)
        logr = self._model.prior.log_prob(x)
        logq = logr + minus_logJ
        return logq


# =============================================================================
class Fitter:
    """A class for training a given model.

    Parameters
    ----------
    model : An instance of Model
    """

    def __init__(self, model):
        self._model = model

        self.train_history = dict(loss=[], logqp=[], logz=[], ess=[])

        self.train_metadata = dict(n_hits=1, print_time=True)

        self.hyperparam = dict(lr=0.001, weight_decay=0.01)

        self.checkpoint_dict = dict(
            display=False,
            print_stride=100,
            print_extra_func=None,
            save_epochs=[],
            save_fname_func=None
            )

    def __call__(self,
            n_epochs=1000,
            batch_size=64,
            optimizer_class=torch.optim.AdamW,
            scheduler=None,
            loss_fn=None,
            hyperparam={},
            train_metadata={},
            checkpoint_dict={}
            ):

        """Fit the model; i.e. train the model.

        Parameters
        ----------
        n_epochs : int
            Number of epochs of training

        batch_size : int
            Size of samples used at each epoch

        optimizer_class : optimization class, optional
            By default is set to torch.optim.AdamW, but can be changed.

        scheduler : scheduler class, optional
            By default no scheduler is used

        loss_fn : None or function, optional
            The default value is None, which translates to using KL divergence

        hyperparam : dict, optional
            Can be used to set hyperparameters like the learning rate and decay
            weights

        train_metadata : dict, optional
            Can be used to set metadata, e.g. number of hits with one batch of
            data

        checkpoint_dict : dict, optional
            Can be set to control the displayed/printed results
        """
        self.train_metadata.update(train_metadata)
        self.hyperparam.update(hyperparam)
        self.checkpoint_dict.update(checkpoint_dict)

        self.loss_fn = Fitter.calc_kl_mean if loss_fn is None else loss_fn

        self.optimizer = optimizer_class(
                self._model.net_.parameters(), **self.hyperparam
                )

        self.scheduler = None if scheduler is None else scheduler(self.optimizer)

        return self.train(n_epochs, batch_size)

    def train(self, n_epochs, batch_size, train_metadata={}):
        """Train the model.

        Parameters
        ----------
        n_epochs : int
            Number of epochs of training

        batch_size : int
            Size of samples used at each epoch

        **Note**: this method is meant to be called by `__call__`,
        but it can be called directly subject to `__call__`
        being called at least once.
        """
        self.train_metadata.update(train_metadata)
        self.train_metadata.update(dict(batch_size=batch_size))
        last_epoch = len(self.train_history["loss"]) + 1
        n_hits = self.train_metadata['n_hits']
        T1 = time.time()
        for epoch in range(last_epoch, last_epoch + n_epochs):
            if (epoch - 1) % n_hits != 0:
                continue
            loss, logqp = self.step()
            self.checkpoint(epoch, logqp)
            if self.scheduler is not None:
                self.scheduler.step()
        T2 = time.time()
        if self.train_metadata['print_time'] and n_epochs > 0:
            print(f"({loss.device}) Time = {T2 - T1:.3g} sec.")

    def step(self):
        """Perform a train step with one batch of inputs"""
        net_ = self._model.net_
        prior = self._model.prior
        action = self._model.action
        batch_size = self.train_metadata['batch_size']
        n_hits = self.train_metadata['n_hits']

        x, logr = prior.sample_(batch_size)
        for _ in range(n_hits):
            y, logJ = net_(x)
            logq = logr - logJ
            logp = -action(y)
            loss = self.loss_fn(logq, logp)
            self.optimizer.zero_grad()  # clears old gradients from last steps
            loss.backward()
            if torch.isnan(loss):
                print("OOPS: loss is divergent -> no *step* is taken.")
            else:
                self.optimizer.step()
            self.train_history['loss'].append(loss.item())

        return loss, logq - logp

    def checkpoint(self, epoch, logqp, n_hits=1):
        print_stride = self.checkpoint_dict['print_stride']
        if epoch == 1 or ((epoch + n_hits - 1) % print_stride) == 0:
            self._append_to_train_history(logqp)
            self.print_fit_status(epoch + n_hits - 1)
            if self.checkpoint_dict['display']:
                self.live_plot_handle.update(self.train_history)
        save_epochs = self.checkpoint_dict['save_epochs']
        save_fname_func = self.checkpoint_dict['save_fname_func']
        if epoch in save_epochs:
            torch.save(self._model.net_, save_fname_func(epoch))

    @staticmethod
    def calc_kl_mean(logq, logp):
        """Return Kullback-Leibler divergence estimated from logq and logp"""
        return (logq - logp).mean()  # KL, assuming samples from q

    @staticmethod
    def calc_kl_var(logq, logp):
        return (logq - logp).var()

    @staticmethod
    def calc_direct_kl_mean(logq, logp):
        r"""Return *direct* KL mean, which is defined as

        .. math::

           \frac{\sum \frac{p}{q} (\log(\frac{p}{q}) + logz)}{\sum \frac{p}{q}}

        where

        .. math::

           logz = \log( \sum(frac{p}{q}) / N)

        wbere N is the number of samples. The direct KL means is invariant
        under scaling p and/or q.
        """
        logpq = logp - logq
        logz = torch.logsumexp(logpq, dim=0) - np.log(logp.shape[0])
        logpq = logpq - logz  # p is now normalized
        p_by_q = torch.exp(logpq)
        return (p_by_q * logpq).mean()

    @staticmethod
    def calc_kl_mean_includelogz(logq, logp):
        logqp = logq - logp
        logz = torch.logsumexp(-logqp, dim=0) - np.log(logp.shape[0])
        return logqp.mean() + logz

    @staticmethod
    def calc_minus_logz(logq, logp):
        logz = torch.logsumexp(logp - logq, dim=0) - np.log(logp.shape[0])
        return -logz

    @staticmethod
    def calc_ess(logq, logp):
        """ESS: effective sample size"""
        logqp = logq - logp
        log_ess = 2*torch.logsumexp(-logqp, dim=0) - torch.logsumexp(-2*logqp, dim=0)
        ess = torch.exp(log_ess) / len(logqp)  # normalized
        return ess

    @torch.no_grad()
    def _append_to_train_history(self, logqp):
        # logqp = logq - logp;  more precisely, logqp = log(q) - log(p * z)
        logz = estimate_logz(logqp, method='jackknife')  # returns (mean, std)
        ess = self.calc_ess(logqp, 0)
        logqp = (logqp.mean().item(), logqp.std().item())
        self.train_history['logqp'].append(logqp)
        self.train_history['logz'].append(logz)
        self.train_history['ess'].append(ess)

    def print_fit_status(self, epoch):
        mydict = self.train_history
        loss = mydict['loss'][-1]
        logqp_mean, logqp_std = mydict['logqp'][-1]
        logz_mean, logz_std = mydict['logz'][-1]
        # We now incorporate the effect of estimated log(z) to mean of log(q/p)
        adjusted_logqp_mean = logqp_mean + logz_mean
        ess = mydict['ess'][-1]

        if epoch == 1:
            print(f"({ess.device}) Training progress:")
            print(f"({ess.device}) Epoch | loss | log(z) | log(q/p)"
                  + " with contribution from log(z)"
                  + "; mean & error from samples in a batch:"
                  )
        str_ = "({4}) Epoch {0} | loss = {1} | log(z) = {2} | log(q/p) = {3}".format(
                epoch,
                "%g" % loss,
                fmt_val_err(logz_mean, logz_std, err_digits=2),
                fmt_val_err(adjusted_logqp_mean, logqp_std, err_digits=2),
                ess.device
                )
        str_ += f" | ess = {ess:g}"

        if self.checkpoint_dict['print_extra_func'] is not None:
            str_ += self.checkpoint_dict['print_extra_func'](epoch)

        print(str_)


# =============================================================================
@torch.no_grad()
def backward_sanitychecker(
        model, n_samples=5, net_=None, return_details=False
        ):
    """Performs a sanity check on the backward method of networks."""

    if net_ is None:
        net_ = model.net_

    x = model.prior.sample(n_samples)
    y, logJ = net_(x)
    x_hat, log0_hat = net_.backward(y, log0=logJ)

    print("Sanity check is OK if following numbers are zero up to round off:")
    print([torch.sum(torch.abs(x - x_hat)), torch.sum(torch.abs(log0_hat))])

    if return_details:
        return (x, y, x_hat), (logJ, log0_hat)
