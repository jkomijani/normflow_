# Copyright (c) 2021-2022 Javad Komijani

"""This module includes utilities for masking inputs"""


from .._normflowcore import torch, torch_device

import itertools


class Mask:

    def __init__(self, shape=None, parity=0, keepshape=True,
            split_form='even-odd', cat_form='even-odd'
            ):
        # Todo: what about mask_on_fly; would it be faster?
        mask_eo = self.evenodd(shape, parity)
        mask_hh = self.halfhalf(shape, parity)
        get_mask = lambda form: mask_eo if form == 'even-odd' else mask_hh
        self.mask = get_mask(split_form)  # .to(torch_device)
        if not keepshape:
            self.cat_mask = get_mask(cat_form)  # .to(torch_device)
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
    def halfhalf(shape, parity):
        mask = torch.empty(shape, dtype=torch.uint8)
        n = (1+shape[-1])//2  # useful for odd size
        mask[..., :n] = parity
        mask[..., n:] = 1 - parity
        return mask

    def _sameshape_split(self, x):
        return (1 - self.mask) * x, self.mask * x

    def _sameshape_cat(self, x_0, x_1):
        return x_0 + x_1

    def _sameshape_purify(self, x_chnl, channel):
        return (1 - self.mask) * x_chnl if channel == 0 else self.mask * x_chnl

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
        mask = self.cat_mask
        x_shape = (*x_0.shape[:-1], *self.shape)
        x = torch.empty(*x_shape)
        x[..., mask == 0] = x_0
        x[..., mask == 1] = x_1
        return x

    def _anothershape_purify(self, x_chnl, channel):
        return x_chnl
