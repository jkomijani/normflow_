import matplotlib.pyplot as plt
from normflow import seize, np


def action_hist(model,
        batch_size=1024, xlim=(-1000, 1000), hist2d_range=None, note=None
        ):
    """make a two-dimensional histogram to compare the true and model dist."""

    y, logq, logp = model.raw_dist.sample_(batch_size)

    s_eff = -seize(logq)
    s = -seize(logp)

    logqp = s - s_eff

    rho = np.corrcoef(s, s_eff)[0, 1]

    offset = np.mean(s) - np.mean(s_eff)

    fig, axs = plt.subplots(1, 2, dpi=120, figsize=(6, 3))
    plt.subplots_adjust(wspace=0.3)

    axs[0].hist2d(s_eff, s, bins=20, range=hist2d_range)

    axs[1].hist(
            [s - s_eff - offset, - s_eff + np.mean(s_eff), - s + np.mean(s)],
            bins=50, stacked=True, color=['b', 'r', 'g'], alpha=0.5,  density=True,
            label=[r"$\log(q/p)$", r"$\log(q)$", "$\log(p)$"]
            )

    x = np.array(xlim)
    axs[0].plot(x, x + offset, ':', color='w', label='slope 1 fit')

    axs[0].set_xlabel(r'$S_{\mathrm{eff}}(z) \equiv -\log~q(z)$')
    axs[0].set_ylabel(r'$S(z) \equiv -\log~p(z)$')
    axs[0].set_title(fr"$\rho = {rho:.2g}$")
    axs[1].set_title("stacked zero-mean hist")
    axs[0].legend(prop={'size': 6})
    axs[1].legend(prop={'size': 8})
    plt.subplots_adjust(top=0.8)

    if note is not None:
        fig.suptitle(note)
