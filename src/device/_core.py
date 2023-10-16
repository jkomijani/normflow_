# Copyright (c) 2023 Javad Komijani, Elias Nyholm

import torch
import numpy as np
import os
import warnings

from functools import partial
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.multiprocessing.spawn import ProcessException


# =============================================================================
class _DDP(torch.nn.parallel.DistributedDataParallel):
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
        self.nranks = 1
        self.rank = 0

    def to(self, *args, **kwargs):
        self._model.net_.to(*args, **kwargs)
        self._model.prior.to(*args, **kwargs)

    def ddp_wrapper(self, rank, nranks):

        # First, move the model (prior and net_) to the specific GPU
        self._model.prior.to(device=rank, dtype=None, non_blocking=False)
        self._model.net_.to(device=rank, dtype=None, non_blocking=False)

        # Second, wrap the net_ with DDP class
        self._model.net_ = DDP(self._model.net_, device_ids=[rank])

        self.nranks = nranks
        self.rank = rank

    def spawnprocesses(self, fn, nranks,
            master_port=12354, seeds_torch=None, *args, **kwargs
            ):
        """
        fn : function
            function to be run in each spawned process. The first argument of fn
            should be 'model' [normflow.Model], i.e. the 'self._model' of this method.
        nranks : int
            number of processes to spawn.
        master_port : int
            open port for communication between processes. Needs to be set manually if
            mutlitple distributed models are trained concurrently on the same machine.
        *args : optional
            will be passed on to fn
        **kargs : optional
            will be passed on to fn
        """

        seeds_torch = prepare_seeds(nranks, seeds_torch)

        wrapped_fn = DistributedFunc(fn)
        try:
            torch.multiprocessing.spawn(
                partial(wrapped_fn, **kwargs),
                # rank is explicitely passed by spawn as the first argument
                args=(nranks, master_port, seeds_torch, self._model) + tuple(args),
                nprocs=nranks,
                join=True
                )
        except ProcessException as e:
            warnings.warn("Distribution training could not be spawned." \
                    + " If default master port is already in use," \
                    + " try setting a different port with the --port option."
                    )
            raise e


class DistributedFunc:

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, rank, nranks, master_port, seeds_torch, model,
            *args, **kwargs
            ):

        setup_process_group(rank, nranks, master_port=master_port)

        model.device_handler.ddp_wrapper(rank, nranks)

        torch.manual_seed(seeds_torch[rank])

        out = self.fn(model, *args, **kwargs)  # call function

        destroy_process_group()  # clean-up NCCL process

        return out


def setup_process_group(rank, world_size, master_addr='localhost', master_port=12354):
    """Initialize NCCL backend for sharing gradients over devices.

    Parameters
    ----------
    rank: int
        Unique identifier of each process
    world_size: int
        Total number of processses
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(master_port)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    # nccl: NVIDIA Collective Communications Library


def prepare_seeds(nranks, seeds_torch):
    """
    seeds_torch : List[int] or None
        List of seeds for torch random number generator. List should be of
        length nranks.
        If None, seeds will be randomly generated using torch.randint.
    """

    if seeds_torch is None:
        seeds_torch = gen_seed(size=(nranks,))
    else:
        assert len(seeds_torch) == nranks, "Numbers of seeds != nranks"

    return seeds_torch


def gen_seed(size=None):
    # if size is None returns a number otherwise a list
    # at least for numpy seed cannot be larger that 2**32 - 1
    if size is None:
        return torch.randint(2**32 - 1, size=[1]).tolist()[0]
    else:
        return torch.randint(2**32 - 1, size=size).tolist()
