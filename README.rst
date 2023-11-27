normflow
--------
This package contains utilities for the implementation of the method of
normalizing flows as a generative model for lattice field theory.

The method of normalizing flows is a powerful approach in generative modeling
that aims to learn complex probability distributions by transforming samples
from a simple distribution through a series of invertible transformations.
It has found applications in various domains, including generative image
modeling.

The package currently supports scalar theories in any dimension, and we are
actively extending the package to accommodate gauge theories, broadening its
applicability.

In a nutshell, for the method of normalizing flows, one should provide three
essential components:
   - A prior distribution to draw initial samples.
   - A neural network to perform a series of invertible transformations on the
     samples.
   - An action that specifies the target distribution, defining the goal of the
     generative model.

The central high-level class of the package is called `Model`, which can be
instantiated by providing instances of the three objects mentioned earlier for
the prior, the neural network, and the action.

Following the terminology used by scikit-learn, every instance of `Model` is
equipped with a method called `fit` that facilitates the training of the model.
Training involves optimizing the parameters of the neural network to achieve a
transformation that effectively maps the prior distribution to the target
distribution.

Below is a simple example of a scalar theory in zero dimension:

    >>> from normflow import Model
    >>> from normflow.action import ScalarPhi4Action
    >>> from normflow.prior import NormalPrior
    >>> from normflow.nn import DistConvertor_
    >>>
    >>> def make_model():
    >>>     prior = NormalPrior(shape=(1,))
    >>>     action = ScalarPhi4Action(kappa=0, m_sq=-1.2, lambd=0.5)
    >>>     net_ = DistConvertor_(knots_len=10, symmetric=True)
    >>>     model = Model(net_=net_, prior=prior, action=action)
    >>>     return model
    >>>
    >>> model = make_model()
    >>> model.fit(n_epochs=500, batch_size=128)
    >>>
    >>> Epoch: 100 | loss: -0.739484 | ... | accept_rate: 0.82(2)
    >>> Epoch: 200 | loss: -0.897035 | ... | accept_rate: 0.845(9)
    >>> Epoch: 300 | loss: -1.01415  | ... | accept_rate: 0.86(1)
    >>> Epoch: 400 | loss: -0.974459 | ... | accept_rate: 0.897(7)
    >>> Epoch: 500 | loss: -1.05209  | ... | accept_rate: 0.914(9)

After training the model, one can draw samples using an attribute called
`posterior`; to draw :math:`n` samples from the trained distribution, use

    >>> x = model.poseterior.sample(n)

Note that the train distribution is almost never identical to the target
distribution, which is specified by the action.
To generate samples that are correctly drawn from the target distribution,
similar to Markov Chain Monte Carlo (MCMC) simulations,
one can employ a Metropolis accept/reject step and discard some of the first
samples; to this end, one can use

    >>> x = model.mcmc.sample(n)

which draws :math:`n` samples from the trained distribution and applies a
Metropolis accept/reject step to ensure that the samples are correctly drawn.

Moreover, the model has an attribute called `device_handler`, which can be used
to specify the number of GPUs used for training (the default value is one if
any GPU is available).
To this end, one can use:

    >>> def fit_func(model):
    >>>     model.fit(n_epochs=500, batch_size=128)
    >>>
    >>> model.device_handler.spawnprocesses(fit_func, nranks)

where `nranks` specifies the number of GPUs.


| Created by Javad Komijani on 2021
| Copyright (C) 2021-23, Javad Komijani
