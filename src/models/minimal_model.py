#!/usr/bin/env python3

"""This module provides an example of how to use the `normflow` package
by building a small (minimal) model for :math:`\phi^4` theory.

The minimal model contains a layer of `FFTNet_` followed by a layer of
`DistConvertor_`.

The `FFTNet_` itself contains a layer of `SplineNet` with number of parameters
that can be controlled by `knots_len1` to manipulate the PSD of ....

Similarly, the numer of parameters of `DistConvertor_` can be controlled by
`knots_len2`.

Setting `knots_len1` and `knots_len2` to integers less than 2 removes
the `SplineNet_` and `DistConvertor_`, respectively.

(Note that `FFTNet_` has at least two parameters, including the effective
mass parameter.)
"""

from normflow import np, torch, grab, torch_device, float_dtype
from normflow import Model, NormalPrior, ScalarPhi4Action
from normflow import FFTNet_, DistConvertor_, ModuleList_
from normflow import MeanFieldNet_


# =============================================================================
def main(*, lat_shape, m_sq, lambd, kappa=1, a=1, eff_mass2=None,
        knots0_len=10, knots1_len=10, knots2_len=10,
        n_epochs=1000, batch_size=128, seed=None,
        sgnbias=False,
        print_net_=False, net_load_fname=None, net_save_fname=None):

    prior = NormalPrior(shape=lat_shape, seed=seed)

    if eff_mass2 is None:
        eff_mass2 = m_sq if lambd == 0 else 1

    fft_dict = dict(eff_mass2=eff_mass2, eff_kappa=kappa, a=a, knots_len=knots1_len)

    if isinstance(net_load_fname, str):
        net_ = torch.load(net_load_fname, map_location=torch.device(torch_device))
        net_.type(float_dtype)
        net_ = net_.transfer(shape=lat_shape)
    else:
        list_nets_ = []
        if knots0_len > 0 or sgnbias:
            list_nets_.append(
                MeanFieldNet_.build(knots0_len, symmetric=True, sgnbias=sgnbias)
                )
        fftnet_ = FFTNet_.build(lat_shape, **fft_dict)
        list_nets_.append(fftnet_)
        if knots2_len > 1:
            list_nets_.append(DistConvertor_(knots2_len, symmetric=True))
        net_ = ModuleList_(list_nets_)

    if print_net_:
        print(net_)

    action_dict = dict(kappa=kappa, m_sq=m_sq, lambd=lambd, a=a)
    action = ScalarPhi4Action(**action_dict, ndim=len(lat_shape))

    model = Model(net_=net_, prior=prior, action=action)

    print("number of model parameters = ", model.net_.npar())

    SchedulerClass = torch.optim.lr_scheduler.CosineAnnealingLR
    scheduler = lambda optimizer: SchedulerClass(optimizer, n_epochs)

    calc_accept = lambda: "| accept_rate = " + model.mcmc.calc_accept_rate(asstr=True)
    extra_func = lambda epoch: calc_accept() if epoch % 500 == 0 else ""

    checkpoint_dict = dict(print_stride=100, print_extra_func=extra_func)
    hyperparam = dict(lr=0.001, weight_decay=0.)

    model.fit(
            n_epochs=n_epochs,
            batch_size=batch_size,
            scheduler=scheduler,
            hyperparam=hyperparam,
            checkpoint_dict=checkpoint_dict
            )

    print("Estimated infrared mass is", fftnet_.infrared_mass)

    if isinstance(net_save_fname, str):
        torch.save(model.net_, net_save_fname)

    return model


# =============================================================================
if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    add = parser.add_argument

    add("--lat_shape", dest="lat_shape", type=str)
    add("--m_sq", dest="m_sq", type=float)
    add("--lambd", dest="lambd", type=float)
    add("--kappa", dest="kappa", type=float)
    add("--a", dest="a", type=float)
    add("--eff_mass2", dest="eff_mass2", type=float)
    add("--knots0_len", dest="knots0_len", type=int)
    add("--knots1_len", dest="knots1_len", type=int)
    add("--knots2_len", dest="knots2_len", type=int)
    add("--batch_size", dest="batch_size", type=int)
    add("--n_epochs", dest="n_epochs", type=int)
    add("--seed", dest="seed", type=int)
    add("--print_net_", dest="print_net_", type=bool)
    add("--sgnbias", dest="sgnbias", type=bool)
    add("--net_load_fname", dest="net_load_fname", type=str)
    add("--net_save_fname", dest="net_save_fname", type=str)

    args = vars(parser.parse_args())
    none_keys = [key for key, value in args.items() if value is None]
    [args.pop(key) for key in none_keys]
    for key in ["lat_shape"]:
        if key in args.keys():
            args[key] = eval(args[key])
    main(**args)

    # print("usage: python minimal_model.py --lat_shape '(8, 8)' --kappa 0.3 --m_sq -1.2 --lambd 0.5
    # main(lat_shape=(8, 8), kappa=1, m_sq=1/4, lambd=0, knots0_len=0, knots1_len=0, knots2_len=0)
    # main(lat_shape=(8, 8), kappa=0.3, m_sq=-1.2, lambd=0.5, knots1_len=0)
