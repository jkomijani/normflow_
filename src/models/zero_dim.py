#!/usr/bin/env python3


from normflow import np, torch, grab
from normflow import Model, NormalPrior, ScalarPhi4Action
from normflow import DistConvertor_


# =============================================================================
def main(*, m_sq, lambd, knots_len=10, n_epochs=1000, batch_size=1024):

    net_ = DistConvertor_(knots_len, symmetric=True)

    lat_shape = (1, )
    action_dict = dict(kappa=0, m_sq=m_sq, lambd=lambd)
    prior = NormalPrior(shape=lat_shape)
    action = ScalarPhi4Action(**action_dict, ndim=len(lat_shape))

    model = Model(net_=net_, prior=prior, action=action)

    print("number of model parameters = ", npar(model.net_))

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

    return model


def npar(net_):
    count = lambda x: np.product(x)
    return sum([count(p.shape) for p in net_.parameters()])


# =============================================================================
if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    add = parser.add_argument

    add("--m_sq", dest="m_sq", type=float)
    add("--lambd", dest="lambd", type=float)
    add("--kappa", dest="kappa", type=float)
    add("--knots_len", dest="knots_len", type=int)
    add("--batch_size", dest="batch_size", type=int)
    add("--n_epochs", dest="n_epochs", type=int)

    args = vars(parser.parse_args())
    none_keys = [key for key, value in args.items() if value is None]
    [args.pop(key) for key in none_keys]
    main(**args)
    # print("usage: python zero_dim.py --m_sq -1.2 --lambd 0.5 --n_epochs 3000 --knots_len 10 --batch_size 1024")
    # main(m_sq=-1.2, lambd=0.5)
