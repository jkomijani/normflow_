import torch
from typing import Tuple, Union
from warnings import warn


class Conv4d(torch.nn.Module):
    def __init__(self,
                    in_channels: int,
                    out_channels: int,
                    kernel_size: Union[Tuple[int, int, int, int], int],
                    stride: int = 1,
                    padding: Union[Tuple[int, int, int, int], int, str] = 0,
                    padding_mode: str = 'circular',
                    dilation: int = 1,
                    groups: int = 1,
                    bias: bool = True,
                    device=None,
                    dtype=None
                    ) -> None:
        super(Conv4d, self).__init__()

        assert stride == 1, \
            'Strides other than 1 not yet implemented!'
        assert padding_mode in ['circular'], \
            'Padding modes other than circular not yet implemented!'
        assert dilation == 1, \
            'Dilation rate other than 1 not yet implemented!'
        assert groups == 1, \
            'Groups other than 1 not yet implemented!'
        if bias:
            warn('Bias is yet not implemented in the extra dimension, and will produce \
                 results slightly different from a correct implementation.')
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size, kernel_size)
        if isinstance(padding, int):
            padding = (padding, padding, padding, padding)
        if isinstance(padding, tuple):
            assert padding[0] == kernel_size[0] // 2, \
                "Only padding mode 'same' is implemented in the first dimension!"
            padding_3d = (padding[1], padding[2], padding[3])
        elif isinstance(padding, str):
            assert padding == 'same', \
                "Only padding mode 'same' is implemented in the first dimension!"
            padding_3d = 'same'
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.device = device
        self.dtype = dtype

        self.conv3d = torch.nn.Conv3d(in_channels=self.in_channels,
                                        out_channels=self.out_channels*kernel_size[0],
                                        kernel_size=(self.kernel_size[1], self.kernel_size[2], self.kernel_size[3]),
                                        stride=self.stride,
                                        padding=padding_3d,
                                        padding_mode=self.padding_mode,
                                        dilation=self.dilation,
                                        groups=self.groups,
                                        bias=bias,
                                        device=self.device,
                                        dtype=self.dtype)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if len(input.shape) == 5:
            # introduce batch dimension
            input = input.unsqueeze(0)

        assert len(input.shape) == 6, "Input tensor must be of shape (batch_size, in_channels, tensor_size_1, tensor_size_2, tensor_size_3))"

        batch_size, in_channels, tensor_size_1, tensor_size_2, tensor_size_3, tensor_size_4 = input.shape
        kernel_size_1, _, _, _ = self.kernel_size

        # TODO: does transpose.reshape cast to view if one of the dimensions is 1?
        if in_channels == 1: # can shuffle indices and keep tensor contiguous
            input = input.view(batch_size*tensor_size_1, 1, tensor_size_2, tensor_size_3, tensor_size_4)
        else: # need to force contiguous tensor (introduces one extra tensor copy)
            input = input.transpose(1,2).reshape(batch_size*tensor_size_1, in_channels, tensor_size_2, tensor_size_3, tensor_size_4)

        output_3d: torch.Tensor = self.conv3d(input)

        if self.out_channels == 1:
            output_3d = output_3d.view(batch_size, 1, tensor_size_1, kernel_size_1, tensor_size_2, tensor_size_3, tensor_size_4)
        else:
            output_3d = output_3d.view(batch_size, tensor_size_1, self.out_channels, kernel_size_1, tensor_size_2, tensor_size_3, tensor_size_4).transpose(1,2).contiguous()
        
        output = torch.empty((batch_size, self.out_channels, tensor_size_1, kernel_size_1, tensor_size_2, tensor_size_3, tensor_size_4), dtype=output_3d.dtype, device=output_3d.device)
        output[:, :, :, 0] = output_3d[:, :, :, 0]
        for i in range(1, kernel_size_1):
            output[:, :, :, i] = output_3d[:, :, :, i].roll(-i+1, dims=2)
            
        output = torch.sum(output, dim=3)

        assert output.shape == (batch_size, self.out_channels, tensor_size_1, tensor_size_2, tensor_size_3, tensor_size_4)
        return output
