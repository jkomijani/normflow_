# Copyright (c) 2021-2023 Javad Komijani

"""This module contains new neural networks..."""


import torch
import copy
import numpy as np

from abc import abstractmethod, ABC

from ...lib.spline import RQSpline
from ...lib.linalg import neighbor_mean
from .conv4d import Conv4d


class Module(torch.nn.Module, ABC):

    def __init__(self, label=None):
        super().__init__()
        self.label = label

    @abstractmethod
    def forward(self, x):
        pass

    @property
    def activations(self):
        return torch.nn.ModuleDict([
                ['tanh', torch.nn.Tanh()],
                ['leaky_relu', torch.nn.LeakyReLU()],
                ['none', torch.nn.Identity()]
               ])

    def transfer(self, **kwargs):
        return copy.deepcopy(self)


class AvgNeighborPool(Module):
    """Return average of all neighbors"""

    def forward(self, x):
        return neighbor_mean(x, dim=range(1, x.ndim))


class ConvAct(Module):
    """
    As an extension to torch.nn.Conv2d, this network is a sequence of
    convolutional layers with possible hidden layers and activations and other
    dimensions.

    Instantiating this class with the default optional variables is equivalent
    to instantiating torch.nn.Conv2d with following optional varaibles:
    padding = 'same' and padding_mode = 'circular'.

    As an option, one can provide a list/tuple for `hidden_sizes`. Then, one
    must also provide another list/tuple for activations using the option
    `acts`; the lenght of `acts` must be equal to the lenght of `hidden_sizes`
    plus 1 (for the output layer).

    The axes of the input and output tensors are treated as
    :math:`tensor(:, ch, ...)`, where `:` stands for the batch axis,
    `ch` for the channels axis, and `...` for the features axes.

    .. math::

        out(:, ch_o, ...) = bias(ch_o) +
                        \sum_{ch_i} weight(ch_o, ...) \star input(:, ch_i, ...)

    where :math:`\star` is n-dimensional cross-correlation operator acting on
    the features axes. The supported features dinensions are 1, 2, 3, and 4.

    Parameters
    ----------
    in_channels (int):
        Number of channels in the input data
    out_channels (int):
        Number of channels produced by the convolution
    kernel_size (int or tuple):
        Size of the convolving kernel
    conv_dim (int, optional):
        Dimension of the convolving kernel (default is 2)
    hidden_sizes (list/tuple of int, optional):
        Sizes of hidden layers (default is [])
    acts (list/tuple of str, optional):
        Activations after each layer (default is 'none')
    """

    Conv = {1: torch.nn.Conv1d,
            2: torch.nn.Conv2d,
            3: torch.nn.Conv3d,
            4: Conv4d
            }

    def __init__(self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            conv_dim: int = 2,
            hidden_sizes = [],
            acts = ['none'],
            **extra_kwargs  # all other kwargs to pass to torch.nn.Conv?d
            ):

        super().__init__()

        Conv = self.Conv[conv_dim]
        sizes = [in_channels, *hidden_sizes, out_channels]
        assert len(acts) == len(hidden_sizes) + 1

        conv_kwargs = dict(padding='same', padding_mode='circular')
        conv_kwargs.update(extra_kwargs)

        net = []
        for i, act in enumerate(acts):
            net.append(Conv(sizes[i], sizes[i+1], kernel_size, **conv_kwargs))
            net.append(self.activations[act])
        self.net = torch.nn.Sequential(*net)

        # save all inputs so that the can be used later for transfer learning
        conv_kwargs.update(
                dict(in_channels=in_channels, out_channels=out_channels,
                     kernel_size=kernel_size, conv_dim=conv_dim,
                     hidden_sizes=hidden_sizes, acts=acts
                     )
                )
        self.conv_kwargs = conv_kwargs

    def forward(self, x):
        return self.net(x)

    def set_param2zero(self):
        for net in self.net:
            for param in net.parameters():
                torch.nn.init.zeros_(param)

    def transfer(self, scale_factor=1, **extra):
        """
        Returns a copy of the current module if scale_factor is 1.
        Otherwise, uses the input scale_factor to resize the kernel size.
        """
        if scale_factor == 1:
            return copy.deepcopy(self)
        else:
            pass  # change the kernel size as below

        ksize = self.conv_kwargs['kernel_size']  # original kernel size
        ksize = 1 + 2 * round((ksize - 1) * scale_factor/2)  # new kernel size

        conv_kwargs = dict(**self.conv_kwargs)
        conv_kwargs['kernel_size'] = ksize

        new_size = [ksize] * conv_kwargs['conv_dim']
        resize = lambda p: torch.nn.functional.interpolate(p, size=new_size)

        state_dict_conv = {key: resize(value)
                for key, value in self.net[::2].state_dict().items()
                }

        state_dict_acts = {key: value
                for key, value in self.net[1::2].state_dict().items()
                }

        state_dict = dict(**state_dict_conv, **state_dict_acts)

        new_net = self.__class__(**conv_kwargs)
        new_net.net.load_state_dict(state_dict)

        return new_net


class LinearAct(Module):
    """A sequence of linear nets with activations. The output is reshaped as
    multiple channles.

    Parameters
    ----------
    features: int
        Nodes in the input layer
    channels: int
        Channels in the output layer
    hidden_sizes: tuple/list of int
        Nodes in the hidden layers
    acts: None or tuple/list of str
        If tuple/list, its length must be equal to the number of layers,
        and if None uses the default values
    bias: bool
        Whether to use biases in the layers
    """
    def __init__(self, *, features, channels,
            hidden_sizes=[], channels_axis=-1, acts=['none'], bias=True
            ):
        super().__init__()
        self.features = features
        self.channels = channels
        self.channels_axis = channels_axis
        sizes = [features, *hidden_sizes, features * channels]
        Linear = torch.nn.Linear
        net = []
        assert len(acts) == len(hidden_sizes)+1
        for i, act in enumerate(acts):
            net.append(Linear(sizes[i], sizes[i+1], bias=bias))
            net.append(self.activations[act])
        self.net = torch.nn.Sequential(*net)

    def forward(self, x):
        x_reshaped = x.reshape(-1, self.features)
        out_shape = list(x.size())
        out_shape[self.channels_axis] = self.channels
        return self.net(x_reshaped).reshape(*out_shape)


class MultiLinearAct(Module):
    """A sequence of linear nets with activations. The output is reshaped as
    multiple channles.

    Parameters
    ----------
    features: int
        Nodes in the input layer
    channels: int
        Channels in the output layer
    hidden_sizes: tuple/list of int
        Nodes in the hidden layers
    acts: None or tuple/list of str
        If tuple/list, its length must be equal to the number of layers,
        and if None uses the default values
    bias: bool
        Whether to use biases in the layers
    """

    def __init__(self, *, threads, features, channels,
            hidden_sizes=[], channels_axis=-1, acts=['none'], bias=True
            ):
        super().__init__()
        self.threads = threads
        self.features = features
        self.channels = channels
        self.channels_axis = channels_axis
        sizes = [features, *hidden_sizes, features * channels]
        Linear = torch.nn.Linear
        threads_net = []
        assert len(acts) == len(hidden_sizes)+1
        for thread in range(threads):
            net = []
            for i, act in enumerate(acts):
                net.append(Linear(sizes[i], sizes[i+1], bias=bias))
                net.append(self.activations[act])
            threads_net.append(torch.nn.Sequential(*net))
        # Pytorch wouldn't recognize Modules if we save threads_net as a list.
        # Therefore, we save threads_net as a ModuleDict
        self.threads_net = torch.nn.ModuleDict(
                [(str(i), net) for i, net in enumerate(threads_net)]
                )
        # self.threads_net either should be an instance of Module or ModuleDict 

    def forward(self, x):
        threads_axis = 1
        x_reshaped = x.reshape(-1, self.threads, self.features)
        split = lambda z: z.split(tuple([1]*self.threads), dim=threads_axis)
        nets = self.threads_net
        out = [nets[str(i)](x_i) for i, x_i in enumerate(split(x_reshaped))]
        out_shape = list(x.size())
        out_shape[self.channels_axis] = self.channels
        return torch.cat(out, threads_axis).reshape(*out_shape)


class SplineNet(Module):

    def __init__(self, knots_len, xlim=(0, 1), ylim=(0, 1),
            knots_x=None, knots_y=None, knots_d=None,
            spline_shape=[], knots_axis=-1,
            smooth=False, Spline=RQSpline, label='spline', **spline_kwargs
            ):
        """
        Return a neural network for spline interpolation/extrapolation.
        The input `knots_len` specifies the number of knots of the spline.
        In general, the first knot is always at (xlim[0], ylim[0]) and the last
        knot is always at (xlim[1], ylim[1]) and the coordintes of other knots
        are network parameters to be trained, unless one explicitely provides
        `knots_x` and/or `knots_y`.
        Assuming `knots_x` is None, one needs `(knots_len - 1)` parameters to
        specify the `x` position of the knots (with softmax);
        similarly for the `y` position.
        There will be additional `knots_len` parameters to specify the
        derivatives at knots unless `smooth == True`.

        Note that `knots_len` must be at least equal 2.
        Also note that

            SplineNet(2, smooth=True)

        is basically an identity net (although it has two dummy parameters!)

        Can be used as a probability distribution convertor for variables with
        nonzero probability in [0, 1].

        Parameters
        ----------
        knots_x : int
             number of knots of the spline.
        xlim : array-like
             the min and max values for `x` of the knots.
        ylim : tuple
             the min and max values for `y` of the knots.
        .
        .
        .

        spline_shape : array-like
            specifies number of splines organized as a tensor
            (default is [], indicating there is only one spline).

        knots_axis : int
            relevant only if spline_shape is not []
            (default value is -1)
        """
        super().__init__(label=label)

        # knots_len and spline_shape are relevant only if rel_flag is True
        rel_flag = (knots_x is None) or (knots_y is None) or (knots_d is None)

        if rel_flag and knots_len < 2:
            raise Exception("Oops: knots_len can't be less than 2 @ SplineNet")

        self.knots_len = knots_len
        self.knots_x = knots_x
        self.knots_y = knots_y
        self.knots_d = knots_d
        self.spline_shape = spline_shape
        self.knots_axis = knots_axis

        self.Spline = Spline
        self.spline_kwargs = spline_kwargs

        self.softmax = torch.nn.Softmax(dim=0)
        self.softplus = torch.nn.Softplus(beta=np.log(2))
        # we set the beta of Softplus to log(2) so that self.softplust(0)
        # returns 1. With this setting it would be easy to set the derivatives
        # to 1 (with zero inputs).

        init = lambda n: torch.zeros(*spline_shape, n)

        if knots_x is None:
            self.xlim, self.xwidth = xlim, xlim[1] - xlim[0]
            self.weights_x = torch.nn.Parameter(init(knots_len - 1))

        if knots_y is None:
            self.ylim, self.ywidth = ylim, ylim[1] - ylim[0]
            self.weights_y = torch.nn.Parameter(init(knots_len - 1))

        if knots_d is None:
            self.weights_d = None if smooth else torch.nn.Parameter(init(knots_len))

    def forward(self, x):
        spline = self.make_spline()
        if len(self.spline_shape) > 0:
            return spline(x)
        else:
            return spline(x.ravel()).reshape(x.shape)

    def backward(self, x):
        spline = self.make_spline()
        if len(self.spline_shape) > 0:
            return spline.backward(x)
        else:
            return spline.backward(x.ravel()).reshape(x.shape)

    def make_spline(self):
        dim = self.knots_axis
        zero_shape = (*self.spline_shape, 1)
        zero = lambda w: torch.zeros(zero_shape, device=w.device)
        cumsumsoftmax = lambda w: torch.cumsum(self.softmax(w), dim=dim)
        to_coord = lambda w: torch.cat((zero(w), cumsumsoftmax(w)), dim=dim)
        to_deriv = lambda d: self.softplus(d) if d is not None else None

        knots_x = self.knots_x
        if knots_x is None:
            knots_x = to_coord(self.weights_x) * self.xwidth + self.xlim[0]

        knots_y = self.knots_y
        if knots_y is None:
            knots_y = to_coord(self.weights_y) * self.ywidth + self.ylim[0]

        knots_d = self.knots_d
        if knots_d is None:
            knots_d = to_deriv(self.weights_d)

        mydict = {'knots_x': knots_x, 'knots_y': knots_y, 'knots_d': knots_d}

        return self.Spline(**mydict, **self.spline_kwargs)


class NOT_USED_SlidingSplineNet(SplineNet):
    """Similar to SplineNet but the `y` value of the start and end points are
    parameters.

    `logy[0]` and logy[1] are logarithm of `y` of the start and end points,
    respectively.
    """

    def __init__(self, knots_len, *, logy, **kwargs):
        super().__init__(knots_len, **kwargs)
        self.logy = torch.nn.Parameter(logy)

    def forward(self, x):
        a, b = torch.exp(self.logy)
        return a + b * super().forward(x)

    def backward(self, x):
        a, b = torch.exp(self.logy)
        return super().backward((x - a)/b)


class Expit(Module):
    """This can be also called Sigmoid"""

    def forward(self, x):
        return 1/(1 + torch.exp(-x))

    def backward(self, x):
        return Logit().forward(x)


class Logit(Module):
    """This is inverse of Sigmoid"""

    def forward(self, x):
        return torch.log(x/(1 - x))

    def backward(self, x):
        return Expit().forward(x)


class DistConvertor(SplineNet):
    """A probability distribution convertor...

    Steps: pass through Expit, SplineNet, and Logit
    """

    def __init__(self, *args, symmetric=False, **kwargs):
        if symmetric:
            extra = dict(xlim=(0.5, 1), ylim=(0.5, 1), extrap={'left':'anti'})
        else:
            extra = dict(xlim=(0, 1), ylim=(0, 1))
        super().__init__(*args, **kwargs, **extra)
        self.expit = Expit()
        self.logit = Logit()

    def forward(self, x):
        return self.logit(super().forward(self.expit(x)))

    def backward(self, x):
        # Note that logit.backward = expit.forward
        return self.logit(super().backward(self.expit(x)))

    def cdf_mapper(self, x):
        """Useful for mapping the CDF of inputs to the CDF of outputs."""
        # The input `x` is expected to be in range 0 to 1.
        return super().forward(x)
