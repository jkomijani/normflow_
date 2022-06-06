"""This is an auxiliarly module for assembling a network"""


from normflow import np, torch, torch_device, float_dtype, float_tensortype
from normflow.nn import ModuleList_
from normflow.nn import PlanarGaugeModule_, NewPlanarGaugeModule_, GaugeModuleList_
from normflow.util.assembler import NetAssembler


class GaugeNetAssembler:

    def __init__(self, features_shape, *, plaq_handle, matrix_handle, **kwargs):

        self.features_shape = features_shape
        self.matrix_handle = matrix_handle  # to parametrize matrices
        self.plaq_handle = plaq_handle  # to relate plaq to links and vice versa
        self.planar_assemblers = self.make_planar_assemblers(features_shape, **kwargs)

        self.nets_ = []

    def __call__(self):
        return GaugeModuleList_(self.nets_)

    @staticmethod
    def make_planar_assemblers(shape, **kwargs):
        """Make one sub-assembler for each plane mainly because each plane can
        have a different mask shape and we want to use the same mask for each
        plane.
        """
        ndim = len(shape)
        planar_assemblers = {}
        for mu in range(ndim):
            nu = (mu - 1) % ndim
            sub_shape = [ell for ell in shape]
            sub_shape[nu] = sub_shape[nu] // 2
            planar_assemblers[(mu, nu)] = NetAssembler(sub_shape, **kwargs)
        return planar_assemblers

    def add_planar_gauge_module_(self, net_, *, zpmask, **kwargs):
        self.nets_.append(
                PlanarGaugeModule_(
                    net_,
                    zpmask=zpmask,
                    plaq_handle=self.plaq_handle,
                    matrix_handle=self.matrix_handle,
                    **kwargs)
                )

    def add_new_planar_gauge_module_(self, net_, *, zpmask, **kwargs):
        self.nets_.append(
                NewPlanarGaugeModule_(
                    net_,
                    zpmask=zpmask,
                    plaq_handle=self.plaq_handle,
                    matrix_handle=self.matrix_handle,
                    **kwargs)
                )

    def transfer(self, net_, **kwargs):
        print("OOPS: not updated")
        new_nets_ = []
        for net_i_ in net_:
            mu, nu = net_i_.zpmask.mu, net_.zpmask.nu
            mask = self.planar_assembler[(mu, nu)].mask
            new_nets_.append(net_i.transfer(mask=mask, **kwargs))
        return GaugeModuleList_(new_nets_)

    @staticmethod
    def save_network(net_, fname):
        torch.save(net_, fname)

    def load_network(self, fname, **kwargs):
        print("OOPS: not updated")
        net_ = torch.load(fname, map_location=torch.device(torch_device))
        net_.type(float_tensortype)
        net_ = self.transfer(net_, **kwargs)
        return net_
