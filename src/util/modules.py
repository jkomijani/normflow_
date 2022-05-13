# Copyright (c) 2021-2022 Javad Komijani

"""This module contains new neural networks..."""


from .._normflowcore import np, torch
from ..lib.spline import Pade22Spline

import copy


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
            padding_mode='circular', bias=True
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

    def forward(self, x):
        return self.net(x)

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

        new_net = ConvAct(**conv_kwargs)
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
            smooth=False, Spline=Pade22Spline, label='spline', **spline_kwargs
            ):
        """The first knot is always at (xlim[0], ylim[0]) and the last knot is
        always at (xlim[1], ylim[1]). Such a system needs `2 (knots_len - 1)`
        parameters to specify the position of the knots (with softmax).
        There will be additional `knots_len` parameters to specify the
        derivatives at knots unless `smooth == True`.

        Note that `knots_len` must be at least equal 2.
        Also note that

            SplineNet(2, smooth=True)

        is basically an identity net (although it has two dummy parameters!)
        """
        super().__init__(label=label)

        if knots_len < 2:
            raise Exception("Oops: knots_len can't be less that 2 @ SplineNet")

        self.knots_len = knots_len
        self.xlim, self.xwidth = xlim, xlim[1] - xlim[0]
        self.ylim, self.ywidth = ylim, ylim[1] - ylim[0]
        self.ylim = ylim
        self.Spline = Spline
        self.spline_kwargs = spline_kwargs

        self.softmax = torch.nn.Softmax(dim=0)
        self.softplus = torch.nn.Softplus()

        # init = lambda n: self.softmax((2*torch.rand(n) - 1) / n**0.5)
        # initial values set to values corresponding to the identity function
        init1 = lambda n: torch.zeros(n)
        init2 = lambda n: torch.log(torch.exp(torch.ones(n)) - 1)

        self.weights_x = torch.nn.Parameter(init1(knots_len - 1))
        self.weights_y = torch.nn.Parameter(init1(knots_len - 1))
        if smooth:
            self.weights_d = None
        else:
            self.weights_d = torch.nn.Parameter(init2(knots_len))

    def forward(self, x):
        spline = self.make_spline()
        return spline(x.ravel()).reshape(x.shape)

    def backward(self, x):
        spline = self.make_spline()
        return spline.backward(x.ravel()).reshape(x.shape)

    def make_spline(self):
        coord = lambda w: torch.cat((torch.tensor([0]),
                                     torch.cumsum(self.softmax(w), dim=0)))
        deriv = lambda d: self.softplus(d) if d is not None else None
        mydict = {'knots_x': coord(self.weights_x) * self.xwidth + self.xlim[0],
                  'knots_y': coord(self.weights_y) * self.ywidth + self.ylim[0],
                  'knots_d': deriv(self.weights_d)}
        mydict.update(self.spline_kwargs)
        return self.Spline(**mydict)


class SlidingSplineNet(SplineNet):
    """Similar to SplineNet but the of the `y` value of the start and end
    points are parameters.

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
            extra = dict(xlim=(0.5, 1), ylim=(0.5, 1), extrap_left='anti')
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
