"""This is an auxiliarly module for assembling a network"""


from normflow import np, torch, torch_device, float_dtype
from normflow import ModuleList_
from normflow import ShiftBlock_, AffineBlock_, P22SBlock_
from normflow import DistConvertor_, Identity_
from normflow import FFTNet_, MeanFieldNet_, PSDBlock_
from normflow import Mask, ConvAct, LinearAct


class NetAssembler:

    def __init__(self, features_shape=None, mask_keepshape=None):

        self.features_shape = features_shape
        if (features_shape is None) or (mask_keepshape is None):
            self.mask = None
        else:
            self.mask = Mask(features_shape, keepshape=mask_keepshape)
        
        self.nets_ = []

    def __call__(self):
        return ModuleList_(self.nets_)

    def conv(self, *, hidden_sizes, acts, in_channels=1, out_channels=2,
            kernel_size=3, conv_dim=2, padding_mode='circular', bias=True
            ):
        kwargs = dict(
                in_channels=in_channels,
                out_channels=out_channels,
                hidden_sizes=hidden_sizes,
                padding_mode=padding_mode,
                kernel_size=kernel_size,
                conv_dim=conv_dim,
                acts=acts,
                bias=bias
                )
        return ConvAct(**kwargs)

    def linear(self, *, channels, hidden_sizes, acts, **kwargs):
        features = np.product(self.features_shape)//2
        kwargs.update(dict(features=features, channels=channels,
                hidden_sizes=hidden_sizes, acts=acts))
        return LinearAct(**kwargs)

    def affine_(self, net1, net2, **kwargs):
        return AffineBlock_(net1, net2, mask=self.mask, **kwargs)

    def shift_(self, net1, net2, **kwargs):
        return ShiftBlock_(net1, net2, mask=self.mask, **kwargs)

    def spline_(self, net1, net2, width=10, **kwargs):
        return P22SBlock_(net1, net2, mask=self.mask, width=width, **kwargs)

    def fftnet_(self, **kwargs):
        return FFTNet_.build(self.features_shape, **kwargs)

    def meanfieldnet_(self, equivariance=True, **kwargs):
        return MeanFieldNet_.build(symmetric=equivariance, **kwargs)

    def psdblock_(self, *, mfnet_, fftnet_):
        return PSDBlock_(mfnet_=mfnet_, fftnet_=fftnet_)

    def distconvertor_(self, knots_len, equivariance=True, **kwargs):
        return DistConvertor_(knots_len, symmetric=equivariance, **kwargs)

    def identity_(self):
        return Identity_()

    def add_affine_(self, *args, **kwargs):
        self.nets_.append(self.affine_(*args, **kwargs))

    def add_shift_(self, *args, **kwargs):
        self.nets_.append(self.shift_(*args, **kwargs))

    def add_spline_(self, *args, **kwargs):
        self.nets_.append(self.spline_(*args, **kwargs))

    def add_fftnet_(self, **kwargs):
        self.nets_.append(self.fftnet_(**kwargs))

    def add_meanfieldnet_(self, *args, **kwargs):
        self.nets_.append(self.meanfieldnet_(*args, **kwargs))

    def add_psdblock_(self, *args, **kwargs):
        self.nets_.append(self.psdblock_(*args, **kwargs))

    def add_distconvertor_(self, *args, **kwargs):
        self.nets_.append(self.distconvertor_(*args, **kwargs))

    def add_identity_(self):
        self.nets_.append(self.indentity_())

    @staticmethod
    def save_network(net_, fname):
        torch.save(net_, fname)

    def load_network(self, fname, **kwargs):
        net_ = torch.load(fname, map_location=torch.device(torch_device))
        net_.type(float_dtype)
        net_ = net_.transfer(mask=self.mask, **kwargs)
        return net_
