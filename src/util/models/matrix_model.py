#!/usr/bin/env python3


from normflow import np, torch, Model
from normflow.nn import MultiChannelModule_
from normflow.nn import MatrixModule_, UnityDistConvertor_, PhaseDistConvertor_
from normflow.action import MatrixAction
from normflow.prior import SUnPrior
from normflow.lib.eig_handle import SUnMatrixParametrizer
from normflow.lib.eig_handle import SU2MatrixParametrizer, SU3MatrixParametrizer

import normflow


# =============================================================================
def main(n=3, beta=1, knots_len=4, n_epochs=1000, batch_size=1024, lr=0.01,
        lat_shape=(1,), func = lambda x: torch.real(x), specialsu3=True,
        only_thetacosphi=False,
        seed=None):

    action = MatrixAction(beta=beta, func=func)

    prior = SUnPrior(n=n, shape=lat_shape, seed=seed)

    par0_net_ = UnityDistConvertor_(knots_len)
    if n == 2:
        par_net_ = par0_net_
        matrix_handle = SU2MatrixParametrizer()
    elif n == 3:
        if specialsu3:
            if only_thetacosphi:
                par1_net_ = normflow.nn.scalar.modules_.Identity_()
            else:
                par1_net_ = PhaseDistConvertor_(knots_len, symmetric=True)
            par_net_ = MultiChannelModule_([par0_net_, par1_net_])
            matrix_handle = SU3MatrixParametrizer()
        else:
            # par0_net_ = PhaseDistConvertor_(knots_len)
            par1_net_ = PhaseDistConvertor_(knots_len)
            # par_net_ = MultiChannelModule_([par0_net_, par1_net_])
            par_net_ = MultiChannelModule_([par1_net_, par1_net_])
            matrix_handle = SUnMatrixParametrizer()

    net_ = MatrixModule_(par_net_, matrix_handle=matrix_handle)

    model = Model(net_=net_, prior=prior, action=action)

    print("number of model parameters =", model.net_.npar)

    calc_accept = lambda: "| accept_rate = " + model.mcmc.calc_accept_rate(asstr=True)
    extra_func = lambda epoch: calc_accept() if epoch % 500 == 0 else ""

    checkpoint_dict = dict(print_stride=100, print_extra_func=extra_func)
    hyperparam = dict(lr=lr, weight_decay=0.)

    model.fit(
            n_epochs=n_epochs,
            batch_size=batch_size,
            hyperparam=hyperparam,
            checkpoint_dict=checkpoint_dict
            )

    return model


# =============================================================================
if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    add = parser.add_argument

    add("-n", dest="n", type=int)
    add("--beta", dest="beta", type=float)
    add("--knots_len", dest="knots_len", type=int)
    add("--batch_size", dest="batch_size", type=int)
    add("--n_epochs", dest="n_epochs", type=int)
    add("--lat_shape", dest="lat_shape", type=str)
    add("--specialsu3", dest="specialsu3", type=bool)

    args = vars(parser.parse_args())
    none_keys = [key for key, value in args.items() if value is None]
    [args.pop(key) for key in none_keys]
    for key in ["lat_shape"]:
        if key in args.keys():
            args[key] = eval(args[key])
    main(**args)

    # print("usage: python matrix_model.py -n 2 --beta 5 --n_epochs 1000 --batch_size 1024 --knots_len 4")
