# Copyright (c) 2023 Javad Komijani, Elias Nyholm

import torch
import os
from functools import partial


# =============================================================================
class DDP(torch.nn.parallel.DistributedDataParallel):
    # After wrapping a Module with DistributedDataParallel, the attributes of
    # the module (e.g. custom methods) became inaccessible. To access them,
    # a workaround is to use a subclass of DistributedDataParallel as here.
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


# =============================================================================
class ModelDeviceHandler:

    def __init__(self, model):
        self._model = model

    def to(self, *args, **kwargs):
        self._model.net_.to(*args, **kwargs)
        self._model.prior.to(*args, **kwargs)

    def distributedto(self, rank, nranks=1, dtype=None, non_blocking=False):
        if nranks > 1:
            # initialize NCCL backend for sharing gradients over devices
            os.environ['MASTER_ADDR'] = 'localhost'
            # we might want to use more dynamic port choice
            os.environ['MASTER_PORT'] = '12354'
            torch.distributed.init_process_group("nccl", rank=rank, world_size=nranks)

        self._model.net_.to(device=rank, dtype=dtype, non_blocking=non_blocking)
        self._model.prior.to(device=rank, dtype=dtype, non_blocking=non_blocking)

    def makesuredistributed(self, rank):
        net_ = self._model.net_

        if isinstance(net_[0], torch.nn.parallel.DistributedDataParallel):
            pass
        elif isinstance(net_, torch.nn.ModuleList):
            net_type = type(net_)
            distributed_net_ = [DDP(subnet_, device_ids=[rank]) for subnet_ in net_]
            self._model.net_ = net_type(distributed_net_)
        elif isinstance(net_, torch.nn.Module):
            self._model.net_ = DDP(net_, device_ids=[rank])
        else:
            return False

        return True

    def spawnprocesses(self, fn, nranks, *args, **kwargs):
        """
        fn : function
            function to be run in each spawned process. The first three arguments
            of fn should be 'rank' [int], 'nranks' [int] and 'model' [normflow.Model].
        nranks : int
            number of processes to spawn.
        *args : optional
            will be passed on to fn
        **kargs : optional
            will be passed on to fn
        """

        wrapped_fn = DistributedFunc(fn)

        torch.multiprocessing.spawn(
            partial(wrapped_fn, **kwargs),
            # rank is explicitely passed by spawn as the first argument
            args=(nranks, self._model) + tuple(args),
            nprocs=nranks,
            join=True
            )


class DistributedFunc:

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, rank, nranks, model, *args, **kwargs):
        # initialise NCCL and move model to device
        model.device_handler.distributedto(rank, nranks)
        model.device_handler.makesuredistributed(rank)

        # call function
        out = self.fn(model, *args, **kwargs)

        # clean-up NCCL process
        torch.distributed.destroy_process_group()
        return out
