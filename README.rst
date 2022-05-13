normflow
--------
This package contains utilities for implementation of normalizing flows as
a generative model using Pytorch.
The package currently supports scalar theories in 0 to 3 dimensions.
(Pytorch does not support ConvNet in more than 3 dimensions yet.)

Three objects are needed for normalizing flows: a prior distribution to
to draw samples, a neural network to transform the samples, and an action
that specifies the target distribution.

The central high-level class of the package is called `Model`.
To instantiate, one should call `Model` with three inputs for the prior
distribution, the designed neural network, and the action of interest.
Following the termilogy used by scikit-learn, every instance of `Model` is
equipped by a method called `fit` that allows to train the model.

After training the model, one can draw samples using an attribute
called `raw_dist`, which itself is an instance of another class;
simply use `model_instance.raw_dist.sample(n)` to draw `n` samples.
These samples are in general not correctly drawn from the distribution governed
by the given action. In order to generate samples that are correcly sampled
one can use `model_instance.mcmc.sample(n)` to draw `n` samples;
this performs a Metropolis accept/reject on the samples.

The easiest way to use this package is to follow the example provided
in `src/models/minimal_model.py`.

| Created by Javad Komijani on 2021
| Copyright (C) 2021-22, Javad Komijani
