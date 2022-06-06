# Copyright (c) 2021-2022 Javad Komijani


import torch
import numpy as np


if torch.cuda.is_available():
    torch_device = 'cuda'
    float_dtype = torch.float32  # np.float32, single
    float_tensortype = torch.cuda.FloatTensor
else:
    torch_device = 'cpu'
    float_dtype = torch.float64  # np.float64, double
    float_tensortype = torch.DoubleTensor

torch.set_default_tensor_type(float_tensortype)
print(f"torch device: {torch_device};  float dtype: {float_dtype}")


def reset_default_tensor_type(dtype, tensortype, device=torch_device):
    torch.set_default_tensor_type(tensortype)
    global float_dtype, float_tensortype, torch_device
    float_dtype, float_tensortype, torch_device = dtype, float_tensortype, device


def manual_torch_seed(seed):
    torch.manual_seed(seed)


def manual_numpy_seed(seed):
    np.random.seed(seed)
