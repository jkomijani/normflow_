
from numpy import ones, mean
from random import randint

from lib.stats.resampler import Resampler

# all this does is check that if you resample from a set of identical numbers the mean and std of the sample and resample will be exactly equal.

def test_resampler():
    resampler = Resampler()
    sample = randint(1,100)*ones((30))
    sample_mean = mean(sample)
    resample_mean, resample_std = resampler.eval(sample)
    assert (sample_mean == resample_mean) & (resample_std == 0)
