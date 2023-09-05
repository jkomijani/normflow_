# Copyright (c) 2021-2023 Javad Komijani

"""This module includes utilities for masking inputs.

Each mask must have three methods:
    1. split (to partition data to two parts),
    2. cat (to put the partitions together),
    3. purify (to make sure there is no contamination from other partition).
"""

import torch
import itertools

from abc import abstractmethod, ABC


class Mask(torch.nn.Module, ABC):
    """Applies the given mask of 0s and 1s."""

    def __init__(self, **mask_kwargs):
        super().__init__()
        mask = self.make_mask(**mask_kwargs)
        self.register_buffer('_mask', mask)
        self.register_buffer('_c_mask', 1 - mask)
        self.mask_kwargs = mask_kwargs

    def __str__(self):
        return self._mask.__str__()

    def split(self, x):
        return self._mask * x, self._c_mask * x

    def cat(self, x_0, x_1):
        return x_0 + x_1

    def purify(self, x_chnl, channel):
        return x_chnl * (self._mask if channel == 0 else self._c_mask)

    @staticmethod
    @abstractmethod
    def make_mask():
        pass


class EvenOddMask(Mask):
    """Creates an even-odd mask of given shape and parity.

    One can exclude a specific direction by providing a value to `exclude_mu`,
    which is by default None. Then the mask in direction of `exclude_mu` is
    constant.
    """

    @staticmethod
    def make_mask(*, shape, parity=0, exclude_mu=None):
        mask = torch.empty(shape, dtype=torch.uint8)
        for ind in itertools.product(*tuple([range(l) for l in shape])):
            if exclude_mu is None:
                mask[ind] = (1 - parity + sum(ind)) % 2
            else:
                mask[ind] = (1 - parity + sum(ind) - ind[exclude_mu]) % 2
        return mask


class AlongAxesEvenOddMask(Mask):
    """Creates a mask that alternates only in a specific given direction."""

    @staticmethod
    def make_mask(*, shape, parity=0, mu=0):
        mask = torch.empty(shape, dtype=torch.uint8)
        for ind in itertools.product(*tuple([range(l) for l in shape])):
            mask[ind] = (1 - parity + ind[mu]) % 2
        return mask


class DummyMask:

    @staticmethod
    def split(x):
        return x, None

    @staticmethod
    def cat(x, dummy):
        return x

    @staticmethod
    def purify(x, *args, **kwargs):
        return x


class ListSplitMask:

    @staticmethod
    def split(x):
        return x[0], x[1]

    @staticmethod
    def cat(x0, x1):
        return [x0, x1]

    @staticmethod
    def purify(x_chnl, *args, **kwargs):
        return x_chnl


class UnbindStackMask:

    def __init__(self, unbind_axis=1):
        self.unbind_axis = unbind_axis

    def split(self, x):
        return list(torch.unbind(x, self.unbind_axis))

    def cat(self, x0, x1):
        return torch.stack([x0, x1], dim=self.unbind_axis)

    @staticmethod
    def purify(x_chnl, *args, **kwargs):
        return x_chnl


class DoubleMask:

    def __init__(self, *, mask0, mask1):
        self.mask0 = mask0
        self.mask1 = mask1

    def split(self, x):
        x_0, x_1 = self.mask0.split(x)
        x_0, self.x_passive = self.mask1.split(x_0)
        return x_0, x_1

    def cat(self, x_0, x_1):
        x_0 = self.mask1.cat(x_0, self.x_passive)
        return self.mask0.cat(x_0, x_1)

    def purify(self, x_chnl, channel, **kwargs):
        return self.mask1.purify(
                self.mask0.purify(x_chnl, channel, **kwargs),
                channel, **kwargs
                )
