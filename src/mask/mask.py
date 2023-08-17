# Copyright (c) 2021-2022 Javad Komijani

"""This module includes utilities for masking inputs."""

# TODO: it is better to save 1 - mask in addition to mask because of AD

import torch

import itertools


class ListSplitMask:
    """Each mask must have two methods: `split` and `cat` to split and
    concatenate the data according to the mask. Another method called ``purify``
    is needed to make sure the data is zero where it must be zero.
    (The ``purify`` method is used by some classes, but not all.)
    """

    @staticmethod
    def split(x):
        return x[0], x[1]

    @staticmethod
    def cat(x0, x1):
        return [x0, x1]

    @staticmethod
    def purify(x_chnl, *args, **kwargs):
        return x_chnl


class _Unused_SplitMask:
    """Each mask must have two methods: `split` and `cat` to split and
    concatenate the data according to the mask. Another method called ``purify``
    is needed to make sure the data is zero where it must be zero.
    (The ``purify`` method is used by some classes, but not all.)
    """

    def __init__(self, split_axis=1):
        self.split_axis = split_axis

    def split(self, x):
        return list(torch.unbind(x, self.split_axis))

    def cat(self, x0, x1):
        return torch.stack([x0, x1], dim=self.split_axis)

    @staticmethod
    def purify(x_chnl, *args, **kwargs):
        return x_chnl


class Mask(torch.nn.Module):
    """Each mask must have two methods: `split` and `cat` to split and
    concatenate the data according to the mask. Another method called ``purify``
    is needed to make sure the data is zero where it must be zero.
    (The ``purify`` method is used by some classes, but not all.)
    """

    def __init__(self, shape=(2, 2), parity=0, keepshape=True,
            split_form='even-odd', mu=None
            ):
        """
        Parameters
        ----------
        split_form : str
            Can be either of 'even-odd', 'half-half', 'directional_even-odd',
            or 'directional_strip'
        mu : int
            Used only with `split_form` set to directional_even-odd or
            directional_strip
        keepshape : boolean (optional)
            Allows to have different masks for splitting and concatenating
            (default is True)
        """

        super().__init__()

        # Todo: what about mask_on_fly? would it be faster?
        def get_mask():
            if split_form == 'even-odd':
                return self.evenodd(shape, parity)
            elif split_form == 'half-half':
                return self.halfhalf(shape, parity)
            elif split_form == 'directional_even-odd':
                return self.directional_evenodd(shape, parity, mu=mu)
            elif split_form == 'directional_strip':
                return self.directional_strip(shape, parity, mu=mu)

        self.register_buffer('mask', get_mask())
        self.shape = shape
        if keepshape:
            self.split = self._sameshape_split
            self.purify = self._sameshape_purify
            self.cat = self._sameshape_cat
        else:
            self.split = self._anothershape_split
            self.purify = self._anothershape_purify
            self.cat = self._anothershape_cat

    @staticmethod
    def evenodd(shape, parity):
        mask = torch.empty(shape, dtype=torch.uint8)
        for ind in itertools.product(*tuple([range(l) for l in shape])):
            mask[ind] = (sum(ind) + parity) % 2
        return mask

    @staticmethod
    def directional_evenodd(shape, parity, *, mu):
        mask = torch.empty(shape, dtype=torch.uint8)
        for ind in itertools.product(*tuple([range(l) for l in shape])):
            mask[ind] = (sum(ind) + parity - ind[mu]) % 2
        return mask

    @staticmethod
    def directional_strip(shape, parity, *, mu):
        mask = torch.empty(shape, dtype=torch.uint8)
        for ind in itertools.product(*tuple([range(l) for l in shape])):
            mask[ind] = (parity + ind[mu]) % 2
        return mask

    @staticmethod
    @staticmethod
    def halfhalf(shape, parity):
        mask = torch.empty(shape, dtype=torch.uint8)
        n = (1 + shape[-1]) // 2  # useful for odd size
        mask[..., :n] = parity
        mask[..., n:] = 1 - parity
        return mask

    def _sameshape_split(self, x):
        return (1 - self.mask) * x, self.mask * x

    def _sameshape_cat(self, x_0, x_1):
        return x_0 + x_1

    def _sameshape_purify(self, x_chnl, channel, zero2one=False):
        if not zero2one:
            if channel == 0:
                return (1 - self.mask) * x_chnl
            else:
                return self.mask * x_chnl
        else:
            if channel == 0:
                return (1 - self.mask) * x_chnl + self.mask
            else:
                return self.mask * x_chnl + (1 - self.mask)

    def _anothershape_split(self, x):
        # Input's shape: x.shape is [..., self.mask.shape]
        # Two outputs' shape : x_i.shape is [..., product(self.mask.shape)/2]
        mask = self.mask
        reshape_size = (*x.shape[:-mask.dim()], -1)
        return (
                torch.masked_select(x, mask == 0).reshape(*reshape_size),
                torch.masked_select(x, mask == 1).reshape(*reshape_size)
               )

    def _anothershape_cat(self, x_0, x_1):
        # Two inputs' shape : x_i.shape is [..., product(self.mask.shape)/2]
        # Output's shape: x.shape is [..., self.mask.shape]
        mask = self.mask
        x_shape = (*x_0.shape[:-1], *self.shape)
        x = torch.empty(*x_shape)
        x[..., mask == 0] = x_0
        x[..., mask == 1] = x_1
        return x

    def _anothershape_purify(self, x_chnl, channel, zero2one=False):
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
