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
import base64
import io
import time
import copy

import numpy as np

from ._mcmc import MCMCSampler, BlockedMCMCSampler
from ._generic import Resampler
from ._generic import grab, estimate_logz, fmt_value_err


# =============================================================================
if torch.cuda.is_available():
    torch_device = 'cuda'
    float_dtype = torch.cuda.FloatTensor  # np.float32 # single
    # float_dtype = torch.cuda.DoubleTensor  # np.float64 # double
else:
    torch_device = 'cpu'
    float_dtype = torch.DoubleTensor  # np.float64 # double
torch.set_default_tensor_type(float_dtype)
print(f"torch device: {torch_device}")


def reset_default_tensor_type(dtype, device=torch_device):
    torch.set_default_tensor_type(dtype)
    global float_dtype, torch_device
    float_dtype, torch_device = dtype, device


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

        self.raw_dist = RawDistribution(self)
        self.mcmc = MCMCSampler(self)
        self.blocked_mcmc = BlockedMCMCSampler(self)

    def transform(self, x):
        return self.net_(x)[0]

    def _set_propagate_density(self, propagate_density):
        """For tests...."""
        self.net_._set_propagate_density(propagate_density)
        self.prior._set_propagate_density(propagate_density)
        self.action._set_propagate_density(propagate_density)


class RawDistribution:
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
    def sample(self, batch_size=1):
        """Return `batch_size` samples."""
        return self._model.net_(self._model.prior.sample(batch_size))[0]

    @torch.no_grad()
    def sample_(self, batch_size=1):
        """Return `batch_size` samples along with corresponding `log(q/p)`."""
        x = self._model.prior.sample(batch_size)
        y, logJ = self._model.net_(x)
        logq = self._model.prior.log_prob(x) - logJ
        logp = -self._model.action(y)  # logp is log(p * z)
        return y, logq - logp

    @torch.no_grad()
    def serial_sample_generator(self, n_samples, batch_size=128):
        """Yield samples one by one"""
        for i in range(n_samples):
            ind = i % batch_size  # the index of the batch
            if ind == 0:
                y, logqp = self.sample_(batch_size)  # logqp is logq - logp
            yield y[ind].unsqueeze(0), logqp[ind].unsqueeze(0)

    def log_prob(self, y, action_logz=0):
        """Returns log probability up to an additive constant."""
        return -self._model.action(y) - action_logz


# =============================================================================
class Fitter:
    """A class for training a given model.

    Parameters
    ----------
    model : An instance of Model
    """

    def __init__(self, model):
        self._model = model

        self.train_history = dict(loss=[], logqp=[], logz=[])

        self.train_metadata = dict(n_hits=1, print_time=True)

        self.hyperparam = dict(lr=0.001, weight_decay=0.01)

        self.checkpoint_dict = dict(
            display=False,
            print_stride=100,
            print_extra_func=lambda *args: '',
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
        self.train_metadata.update(train_metadata)
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
        last_epoch = len(self.train_history["loss"]) + 1
        if self.train_metadata['print_time']:
            print("Time = {:.3g} sec.".format(T2 - T1))

    def step(self):
        """Perform a train step with one batch of inputs"""
        net_ = self._model.net_
        prior = self._model.prior
        action = self._model.action
        batch_size = self.train_metadata['batch_size']
        n_hits = self.train_metadata['n_hits']

        x = prior.sample(batch_size)
        logq = prior.log_prob(x)
        for _ in range(n_hits):
            logp = -action(*net_(x))  # logJ is absorbed in logp
            loss = self.loss_fn(logq, logp)
            self.optimizer.zero_grad()  # clears old gradients from last steps
            loss.backward()
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
        """An example with extremely small tunneling (like $m^2 = -1$ and
        $\lambda = 0.01$ shows that the only method that gives a reasonable
        result is this one; thus, ignore other methods such as calc_kl_var
        and...
        """
        return (logq - logp).mean()  # reverse KL, assuming samples from q

    @staticmethod
    def calc_kl_var(logq, logp):
        return (logq - logp).var()

    @staticmethod
    def calc_kl_mean_var(logq, logp):
        return (logq - logp).mean()**2 + (logq - logp).var()

    @staticmethod
    def calc_direct_kl_mean(logq, logp):
        logpq = logp - logq
        p_by_q = torch.exp(logpq - logpq.mean())
        return (p_by_q * logpq).mean()

    @staticmethod
    def calc_kl_mean_includelogz(logq, logp):
        logqp = logq - logp
        logz = torch.logsumexp(-logqp, dim=0) - np.log(logp.shape[0])
        return logqp.mean() + logz

    @staticmethod
    def calc_minus_logz(logq, logp):
        logz = torch.logsumexp(logp -logq, dim=0) - np.log(logp.shape[0])
        return -logz

    @torch.no_grad()
    def _append_to_train_history(self, logqp):
        # logqp = logq - logp;  more precisely, logqp = log(q) - log(p * z)
        logz = estimate_logz(logqp)  # returns (mean, std)
        logqp = (logqp.mean().item(), logqp.std().item())
        self.train_history['logqp'].append(logqp)
        self.train_history['logz'].append(logz)

    def print_fit_status(self, epoch):
        mydict = self.train_history
        loss = mydict['loss'][-1]
        logqp_mean, logqp_std = mydict['logqp'][-1]
        logz_mean, logz_std = mydict['logz'][-1]
        # We now incorporate the effect of estimated log(z) to mean of log(q/p)
        adjusted_logqp_mean = logqp_mean + logz_mean
        if epoch == 1:
            print("Training progress:")
            print("Epoch | loss | log(z) | log(q/p) with contribution from log(z)"
                  + "; mean & error from samples in a batch:"
                  )
        str_ = "Epoch {0} | loss = {1} | log(z) = {2} | log(q/p) = {3}".format(
                epoch,
                "%g" % loss,
                fmt_value_err(logz_mean, logz_std, err_digits=2),
                fmt_value_err(adjusted_logqp_mean, logqp_std, err_digits=2),
                )
        print(str_, self.checkpoint_dict['print_extra_func'](epoch))


# =============================================================================
class Module_(torch.nn.Module):
    """A prototype class: like a `torch.nn.Module` except for the `forward`
    and `backward` methods that handle the Jacobians of the transformation.
    We use trailing underscore to denote the neworks in which the `forward`
    and `backward` methods handle the Jacobians of the transformation.
    """

    # We are going to call sum_density with prefix self, so you need to include
    # self as the first argument. 
    sum_density = lambda self, x: torch.sum(x, dim=list(range(1, x.dim())))

    _propagate_density = False  # for test

    def __init__(self, label=None):
        super().__init__()
        self.label = label

    def forward(self, x, log0=0):
        pass

    def backward(self, x, log0=0):
        pass

    def transfer(self, **kwargs):
        return copy.deepcopy(self)

    @staticmethod
    def _set_propagate_density(propagate_density):
        """Define a lambda function for (not) summing up a tensor over all axes
        except the batch axis."""
        if propagate_density:
            func = lambda dummy, x: x
        else:
            func = lambda dummy, x: torch.sum(x, dim=list(range(1, x.dim())))
        Module_.sum_density = func
        # because sum_density is a method, the first input would be `self`
        # or any dummy variable
        Module_._propagate_density = propagate_density


# =============================================================================
class ModuleList_(torch.nn.ModuleList):
    # Do NOT build any child class of ModuleList_!
    # pickle has a problem with saving/loading child classes of ModuleList_;
    # it saves their instances as ModuleList_ instances!
    """Like `torch.nn.ModuleList` except for the `forward` and `backward`
    methods that handle the Jacobians of the transformation.
    We use trailing underscore to denote the neworks in which the `forward`
    and `backward` methods handle the Jacobians of the transformation.

    Parameters
    ----------
    nets_ : instance of Module_ or ModuleList_
    """

    def __init__(self, nets_, label=None):
        super().__init__(nets_)
        self.label = label
        self._groups = None

    def forward(self, x, log0=0):
        for net_ in self:
            x, log0 = net_.forward(x, log0)
        return x, log0

    def backward(self, x, log0=0):
        for net_ in self[::-1]:
            x, log0 = net_.backward(x, log0)
        return x, log0

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def parameters(self):
        if self._groups is None:
            return super().parameters()
        else:
            params_list = []
            sum_ = lambda x: sum(x, start=[])
            for grp in self._groups:
                par = sum_([list(self[k].parameters()) for k in grp['ind']])
                params_list.append(dict(params=par, **grp['hyper']))
            return params_list

    def setup_groups(self, groups=None):
        """If group is not None, it must be a list of dicts. e.g. as
        groups = [{'ind': [0, 1], 'hyper': dict(weight_decay=1e-4)},
                  {'ind': [2, 3], 'hyper': dict(weight_decay=1e-2)}]
        """
        self._groups = groups

    def npar(self):
        count = lambda x: np.product(x)
        return sum([count(p.shape) for p in super().parameters()])

    def hack(self, x, log0=0):
        """Similar to the forward method, except that returns the output of
        middle blocks too; useful for examining effects of each block.
        """
        stack = [(x, log0)]
        for net_ in self:
            x, log0 = net_.forward(x, log0)
            stack.append((x, log0))
        return stack

    def transfer(self, **kwargs):
        return ModuleList_([net_.transfer(**kwargs) for net_ in self])

    def get_weights_blob(self):
        serialized_model = io.BytesIO()
        torch.save(self.state_dict(), serialized_model)
        return base64.b64encode(serialized_model.getbuffer()).decode('utf-8')

    def set_weights_blob(self, blob):
        weights = torch.load(
                io.BytesIO(base64.b64decode(blob.strip())),
                map_location=torch.device('cpu'))
        self.load_state_dict(weights)
        if torch_device == 'cuda':
            self.cuda()

    def freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = True

    @staticmethod
    def _set_propagate_density(arg):
        Module_._set_propagate_density(arg)


# =============================================================================
