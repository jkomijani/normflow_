# Copyright (c) 2021-2022 Javad Komijani

"""This module is for introducing priors..."""


from .._normflowcore import np, torch


class Prior:
    """A template class to initiate a prior distribution."""

    def __init__(self, dist, seed=None, propagate_density=False):
        self.dist = dist
        Prior.manual_seed(seed)
        self._propagate_density = propagate_density  # for test

    def sample(self, batch_size=1):
        return self.dist.sample((batch_size,))

    def log_prob(self, x):
        if self._propagate_density:
            return self.dist.log_prob(x)
        else:
            axes = range(1, len(x.shape))
            return torch.sum(self.dist.log_prob(x), dim=tuple(axes))

    def _set_propagate_density(self, propagate_density):  # for test
        self._propagate_density = propagate_density

    @staticmethod
    def manual_seed(seed):
        if isinstance(seed, int):
            torch.manual_seed(seed)

    def dof(self):
        return np.product(self.shape)


class UniformPrior(Prior):
    """Creates a uniform distribution parameterized by low and high;
    uniform in [low, hight].
    """

    def __init__(self, low=None, high=None, shape=None, seed=None, **kwargs):
        """If shape is None, low & high must be of similar shape."""
        if shape is not None:
            low = torch.zeros(shape)
            high = torch.ones(shape)
        else:
            shape = low.shape
        dist = torch.distributions.uniform.Uniform(low, high)
        super().__init__(dist, seed, **kwargs)
        self.shape = shape


class NormalPrior(Prior):
    """Creates a normal distribution parameterized by loc and scale."""

    def __init__(self, loc=None, scale=None, shape=None, seed=None, **kwargs):
        """If shape is None, loc & scale must be of similar shape."""
        if shape is not None:
            loc = torch.zeros(shape)  # i.e. mean
            scale = torch.ones(shape)  # i.e. sigma
        else:
            shape = loc.shape
        dist = torch.distributions.normal.Normal(loc, scale)
        super().__init__(dist, seed, **kwargs)
        self.shape = shape

    def setup_blockupdater(self, block_len):
        chopped_prior = NormalPrior(shape=block_len)
        self._blockupdater = BlockUpdater(chopped_prior, block_len)

    def update(self, x, block_ind):
        """In-place updater"""
        self._blockupdater(x, block_ind)


class NormalPriorWithOutlier(NormalPrior):

    def __init__(self, outlier_factor=3, outlier_prob=0.05, **kwargs):
        super().__init__(**kwargs)

        self.outlier_prior = NormalPrior(
                loc=self.dist.loc,
                scale=self.dist.scale * outlier_factor
                )
        self.outlier_prob = outlier_prob

    def sample(self, batch_size=1):
        size0 = int(batch_size * self.outlier_prob)
        size1 = batch_size - size0
        sampler0 = self.outlier_prior.dist.sample  # sample from outlier
        sampler1 = self.dist.sample  # sample from NormalPrior
        if size0 > 0:
            return torch.cat((sampler0((size0,)), sampler1((size1,))), dim=0)
        else:
            return sampler1((size1,))


class BlockUpdater:

    def __init__(self, chopped_prior, block_len):
        self.block_len = block_len
        self.chopped_prior = chopped_prior

    def __call__(self, x, block_ind):
        """In-place updater"""
        batch_size = x.shape[0]
        view = x.view(batch_size, -1, self.block_len)
        view[:, block_ind] = self.chopped_prior.sample(batch_size)
