# Copyright (c) 2021-2023 Javad Komijani

"""This module includes utilities for masking inputs.

Each mask must have three methods:
    1. split (to partition data to two parts),
    2. cat (to put the partitions together),
    3. purify (to make sure there is no contamination from other partition).
"""

import torch

from .mask import EvenOddMask, AlongAxesEvenOddMask


class DoubleMask:

    def __init__(self, *, passive_maker_mask, frozen_maker_mask):
        self.passive_maker_mask = passive_maker_mask
        self.frozen_maker_mask = frozen_maker_mask

    def split(self, x):
        x, self._x_passive = self.passive_maker_mask.split(x)
        return self.frozen_maker_mask.split(x)

    def cat(self, x_0, x_1):
        x = self.frozen_maker_mask.cat(x_0, x_1)
        return self.passive_maker_mask.cat(x, self._x_passive)

    def purify(self, x_chnl, channel, **kwargs):
        return self.passive_maker_mask.purify(
                self.frozen_maker_mask.purify(x_chnl, channel, **kwargs),
                0  # 1, which corresponds to self._x_passive is out of access
                )

class GaugeLinksDoubleMask(DoubleMask):

    def __init__(self, *, shape, parity, mu, channels_exist=True):
        # If axis = 1 is the channels axis (e.g., eigenvalues), then mu should
        # be modified accordingly.
        mask0 = EvenOddMask(shape=mask_shape, parity=parity)
        mask1 = AlongAxesEvenOddMask(shape=mask_shape, mu=mu)
        super().__init__(passive_maker_mask=mask0, frozen_maker_mask=mask1)
