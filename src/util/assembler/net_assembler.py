"""This is an auxiliarly module for assembling a network"""


from normflow import np, torch, torch_device, float_dtype, float_tensortype
from normflow.nn import ModuleList_
from normflow.nn import ShiftBlock_, AffineBlock_, RQSplineBlock_
from normflow.nn import U1RQSplineBlock_, SU2RQSplineBlock_, SU3RQSplineBlock_
from normflow.nn import DistConvertor_, Identity_
from normflow.nn import FFTNet_, MeanFieldNet_, PSDBlock_
from normflow.nn import ConvAct, LinearAct
from normflow.mask import Mask, SplitMask


class NetAssembler:

    def __init__(self, features_shape=None, mask_keepshape=None,
            split_mask=False, split_axis=None):

        self.features_shape = features_shape

        if split_mask:
            self.mask = SplitMask(split_axis)
        elif (features_shape is None) or (mask_keepshape is None):
            self.mask = None
        else:
            self.mask = Mask(features_shape, keepshape=mask_keepshape)
        
        self.nets_ = []

    def __call__(self):
        return ModuleList_(self.nets_)

    def conv(self, *, hidden_sizes, acts, in_channels=1, out_channels=2,
            kernel_size=3, conv_dim=2, padding_mode='circular', bias=True,
            set_param2zero=False
            ):
        kwargs = dict(
                in_channels=in_channels,
                out_channels=out_channels,
                hidden_sizes=hidden_sizes,
                padding_mode=padding_mode,
                kernel_size=kernel_size,
                conv_dim=conv_dim,
                acts=acts,
                bias=bias,
                set_param2zero=set_param2zero
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

    def spline_(self, net1, net2, **kwargs):
        return RQSplineBlock_(net1, net2, mask=self.mask, **kwargs)

    def u1_spline_(self, net1, net2, **kwargs):
        mydict = dict(xlim=(-np.pi, np.pi), ylim=(-np.pi, np.pi))
        mydict.update(kwargs)
        return U1RQSplineBlock_(net1, net2, mask=self.mask, **mydict)

    def su2_spline_(self, net1, net2, **kwargs):
        mydict = dict(xlim=(0, 1), ylim=(0, 1))
        mydict.update(kwargs)
        return SU2RQSplineBlock_(net1, net2, mask=self.mask, **mydict)

    def su3_spline_(self, net1, net2, **kwargs):
        mydict = dict(xlims=[(0, 1), (-np.pi, np.pi)], ylims=[(0, 1), (-np.pi, np.pi)], boundaries=['none', 'periodic'])
        mydict.update(kwargs)
        return SU3RQSplineBlock_(net1, net2, mask=self.mask, **mydict)

    def fftnet_(self, **kwargs):
        return FFTNet_.build(self.features_shape, **kwargs)

    def meanfieldnet_(self, zee2sym=True, **kwargs):
        return MeanFieldNet_.build(symmetric=zee2sym, **kwargs)

    def psdblock_(self, *, mfnet_, fftnet_):
        return PSDBlock_(mfnet_=mfnet_, fftnet_=fftnet_)

    def distconvertor_(self, knots_len, zee2sym=True, **kwargs):
        return DistConvertor_(knots_len, symmetric=zee2sym, **kwargs)

    def identity_(self):
        return Identity_()

    def add_affine_(self, *args, **kwargs):
        self.nets_.append(self.affine_(*args, **kwargs))

    def add_shift_(self, *args, **kwargs):
        self.nets_.append(self.shift_(*args, **kwargs))

    def add_spline_(self, *args, **kwargs):
        self.nets_.append(self.spline_(*args, **kwargs))

    def add_u1_spline_(self, *args, **kwargs):
        self.nets_.append(self.phase_spline_(*args, **kwargs))

    def add_su2_spline_(self, *args, **kwargs):
        self.nets_.append(self.su2_spline_(*args, **kwargs))
    
    def add_su3_spline_(self, *args, **kwargs):
        self.nets_.append(self.su3_spline_(*args, **kwargs))
    
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
        net_.type(float_tensortype)
        net_ = net_.transfer(mask=self.mask, **kwargs)
        return net_
