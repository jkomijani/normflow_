# Copyright (c) 2021-2023 Javad Komijani


import torch
import numpy as np


if torch.cuda.is_available():
    torch_device = 'cuda'
    # float_dtype = torch.float32  # np.float32, single
    # float_tensortype = torch.cuda.FloatTensor
    float_dtype = torch.float64  # np.float64, double
    float_tensortype = torch.cuda.DoubleTensor
else:
    torch_device = 'cpu'
    float_dtype = torch.float64  # np.float64, double
    float_tensortype = torch.DoubleTensor

torch.set_default_dtype(float_dtype)
print(f"torch device: {torch_device};  float dtype: {float_dtype}")


def reset_default_dtype_device(dtype=float_dtype, device=torch_device):
    torch.set_default_dtype(dtype)
    torch.set_default_device(device)
    global float_dtype, torch_device
    float_dtype, torch_device = dtype, device


def get_default_dtype_device():
    return float_dtype, torch_device


def manual_torch_seed(seed):
    torch.manual_seed(seed)


def manual_numpy_seed(seed):
    np.random.seed(seed)
