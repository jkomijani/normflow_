# Copyright (c) 2023 Javad Komijani, Elias Nyholm

import torch
import numpy as np
import os
from functools import partial
from torch.multiprocessing.spawn import ProcessException
import warnings


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
    def __init__(self, model, seed_torch=None, seed_np=None):
        self._model = model
        self.nranks = 1
        self.rank = 0

        self.seed_torch = gen_seed() if seed_torch is None else seed_torch
        self.seed_np = gen_seed() if seed_np is None else seed_np

        self.seeds_np = None   # a list of seeds for distributed training
        self.seeds_torch = None

        torch.manual_seed(self.seed_torch)
        np.random.seed(self.seed_np)

    def to(self, *args, **kwargs):
        self._model.net_.to(*args, **kwargs)
        self._model.prior.to(*args, **kwargs)

    def distributedto(self, rank, *, seed_np, seed_torch, nranks=1,
            master_port=12354, dtype=None, non_blocking=False
            ):
        if nranks > 1:
            # initialize NCCL backend for sharing gradients over devices
            os.environ['MASTER_ADDR'] = 'localhost'
            # we might want to use more dynamic port choice
            os.environ['MASTER_PORT'] = str(master_port)
            torch.distributed.init_process_group("nccl", rank=rank, world_size=nranks)

            self.seed_torch = seed_torch
            torch.manual_seed(self.seed_torch)

            self.seed_np = seed_np
            np.random.seed(self.seed_np)

        self._model.net_.to(device=rank, dtype=dtype, non_blocking=non_blocking)
        self._model.prior.to(device=rank, dtype=dtype, non_blocking=non_blocking)

        self.nranks = nranks
        self.rank = rank

    def _makesuredistributed(self, rank):
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

    def spawnprocesses(self, fn, nranks,
            master_port=12354, seeds_np=None, seeds_torch=None, *args, **kwargs
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
        seeds_np : List[int]
            list of seeds for numpy random number generator. List should be of length nranks.
            If None, seeds will be randomly generated using np.random.randint.
        seeds_torch : List[int]
            list of seeds for torch random number generator. List should be of length nranks.
            If None, seeds will be randomly generated using torch.randint.
        *args : optional
            will be passed on to fn
        **kargs : optional
            will be passed on to fn
        """

        self.seeds_torch = gen_seed(size=(nranks,)) if seeds_torch is None else seeds_torch
        self.seeds_np = gen_seed(size=(nranks,)) if seeds_np is None else seeds_np

        if len(self.seeds_np) != nranks or len(self.seeds_torch) != nranks:
            raise ValueError("Number of seeds does not equal nranks.")

        wrapped_fn = DistributedFunc(fn)
        try:
            torch.multiprocessing.spawn(
                partial(wrapped_fn, **kwargs),
                # rank is explicitely passed by spawn as the first argument
                args=(nranks, master_port, self.seeds_np, self.seeds_torch, self._model) + tuple(args),
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

    def __call__(self, rank, nranks, master_port, seeds_np, seeds_torch, model, *args, **kwargs):
        # initialise NCCL and move model to device
        seed_np = seeds_np[rank]
        seed_torch = seeds_torch[rank]

        model.device_handler.distributedto(rank, nranks=nranks,
                seed_np=seed_np, seed_torch=seed_torch, master_port=master_port
                )

        # model.device_handler.makesuredistributed(rank)  ## Do we want this!?

        # call function
        out = self.fn(model, *args, **kwargs)

        # clean-up NCCL process
        torch.distributed.destroy_process_group()
        return out


def gen_seed(size=None):
    # if size is None returns a number otherwise a list
    # at least for numpy seed cannot be larger that 2**32 - 1
    if size is None:
        return torch.randint(2**32 - 1, size=[1]).tolist()[0]
    else:
        return torch.randint(2**32 - 1, size=size).tolist()
