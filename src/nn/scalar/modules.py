# Copyright (c) 2021-2022 Javad Komijani

"""This module contains new neural networks..."""


import torch
import copy
import numpy as np

from ...lib.spline import RQSpline


class Module(torch.nn.Module):

    def __init__(self, label=None):
        super().__init__()
        self.label = label

    @property
    def activations(self):
        return torch.nn.ModuleDict([
                ['tanh', torch.nn.Tanh()],
                ['leaky_relu', torch.nn.LeakyReLU()],
                ['none', torch.nn.Identity()],
                ['dc-sym', DistConvertor(10, symmetric=True)],
                ['dc-asym', DistConvertor(10, symmetric=False)]
               ])

    def transfer(self, **kwargs):
        return copy.deepcopy(self)


class ConvAct(Module):
    """A sequence of conv nets with activations.

    Parameters
    ----------
    in_channels:  int
        Channels in the input layer
    out_channels:
        Channels in the output layer
    hidden_sizes: tuple/list
        Channes in the hidden layers
    acts: tuple/list of str or None
        If tuple/list, its length must be equal to the number of layers,
        and if None uses the default values
    bias: bool
        Whether to use biases in the layers
    """

    def __init__(self, *, in_channels, out_channels, conv_dim,
            hidden_sizes=[], kernel_size=3, acts=['none'],
            padding_mode='circular', bias=True,
            set_param2zero=False
            ):
        super().__init__()
        sizes = [in_channels, *hidden_sizes, out_channels]
        padding_size = (kernel_size // 2)
        conv_kwargs = dict(
                padding=padding_size, padding_mode=padding_mode, bias=bias
                )

        if conv_dim == 1:
            Conv = torch.nn.Conv1d
        elif conv_dim == 2:
            Conv = torch.nn.Conv2d
        elif conv_dim == 3:
            Conv = torch.nn.Conv3d

        net = []
        for i, act in enumerate(acts):
            net.append(Conv(sizes[i], sizes[i+1], kernel_size, **conv_kwargs))
            net.append(self.activations[act])
        self.net = torch.nn.Sequential(*net)

        self.conv_kwargs = dict(
                in_channels=in_channels, out_channels=out_channels,
                conv_dim=conv_dim, hidden_sizes=hidden_sizes,
                kernel_size=kernel_size, acts=acts,
                padding_mode=padding_mode, bias=bias
                )

        if set_param2zero:
            self.set_param2zero()

    def forward(self, x):
        return self.net(x)

    def set_param2zero(self):
        for net in self.net:
            for param in net.parameters():
                torch.nn.init.zeros_(param)

    def transfer(self, scale_factor=1, **extra):
        """Returns a copy of the current module if scale_factor is 1.
        Otherwise, use the input scale_factor to resize the kernel size.
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
            self.device = self.weights_x.device
        else:
            self.device = self.knots_x.device

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
        zero = torch.zeros(zero_shape, device=self.device)
        cumsumsoftmax = lambda w: torch.cumsum(self.softmax(w), dim=dim)
        to_coord = lambda w: torch.cat((zero, cumsumsoftmax(w)), dim=dim)
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
