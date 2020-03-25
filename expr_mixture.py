import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import edgeworth
import time
import pickle

from matplotlib import rc

rc("text", usetex=True)


def run_expr_mixture(
    dens_func_P,
    dens_func_Q,
    log_likelihood_ratio_func,
    num_composition,
    mu_gaussian,
    p,
    use_cornish_fisher=False,
    left=-np.inf,
    right=np.inf,
    title=None,
    ax=None,
    log_scale=False,
    zoom_in=False,
):

    print("$$$$$ n:{}".format(num_composition))
    moments_P, errs = edgeworth.compute_moments(
        log_likelihood_ratio_func, dens_func_P, max_order=4, left=left, right=right
    )
    kappas_P = edgeworth.compute_cumulants(moments_P)

    moments_Q, errs = edgeworth.compute_moments(
        log_likelihood_ratio_func, dens_func_Q, max_order=4, left=left, right=right
    )
    kappas_Q = edgeworth.compute_cumulants(moments_Q)
    print("\t\tCumulants of Q: {}".format(kappas_Q))
    print("\t\tCumulants of P: {}".format(kappas_P))

    sigma_square_P = [kappas_P[1]] * num_composition
    sigma_square_Q = [kappas_Q[1]] * num_composition
    kappa_3_P = [kappas_P[2]] * num_composition
    kappa_3_Q = [kappas_Q[2]] * num_composition
    kappa_4_P = [kappas_P[3]] * num_composition
    kappa_4_Q = [kappas_Q[3]] * num_composition

    mu_f = (
        (moments_Q[0] - moments_P[0]) / np.sqrt(kappas_P[1]) * np.sqrt(num_composition)
    )
    alpha = np.linspace(1e-7, 1 - 1e-7, 100)
    print("\t\tmu_f:{}".format(mu_f))

    start = time.time()
    f_clt = edgeworth.compute_gdp_clt_mixture(alpha, mu_gaussian, p, num_composition)
    clt_time = time.time() - start
    print("\t\tCLT used {} seconds.".format(clt_time))

    f_edgeworth = []
    start = time.time()
    for aa in alpha:
        val = edgeworth.approx_f_edgeworth(
            aa,
            sigma_square_P,
            sigma_square_Q,
            kappa_3_P,
            kappa_3_Q,
            kappa_4_P,
            kappa_4_Q,
            mu_f,
            use_cornish_fisher=use_cornish_fisher,
        )
        f_edgeworth.append(val)
    edgeworth_time = time.time() - start
    f_edgeworth = np.array(f_edgeworth)
    print("\t\tEdgeworth used {} seconds.".format(edgeworth_time))

    epsilon = np.linspace(-6, 6, 100)
    start = time.time()
    f_numerical, f_eps_delta, deltas = edgeworth.approx_f_numerical(
        dens_func_Q,
        log_likelihood_ratio_func,
        num_composition,
        epsilon,
        alpha=alpha,
        asymm=True,
    )
    numerical_time = time.time() - start
    print("\t\tNumerical used {} seconds.".format(numerical_time))

    if num_composition == 1:
        f_true = []
        f_gaussian = edgeworth.compute_gdp_clt(alpha, mu_gaussian)
        for ii in range(len(f_clt)):
            aa = alpha[ii]
            f_true.append((1 - p) * (1.0 - aa) + p * f_gaussian[ii])
    if ax is not None:
        line1, = ax.plot(alpha, f_numerical, linewidth=4, color="k", linestyle="-")
        line2, = ax.plot(alpha, f_edgeworth, linewidth=4, color="r")
        line3, = ax.plot(alpha, f_clt, linewidth=4, color="b", linestyle="--")
        if num_composition == 1:
            line4, = ax.plot(alpha, f_true, color="navy", linestyle="-")
        ax.set_xlabel(r"$\alpha$", fontsize=30)
        ax.set_ylabel(r"$f(\alpha)$", fontsize=30)
        ax.xaxis.set_tick_params(size=18, labelsize=30)
        ax.yaxis.set_tick_params(size=18, labelsize=30)
        if title is not None:
            ax.set_title(title, fontdict={"fontsize": 30})
        if log_scale:
            ax.set_yscale("log")
        ax.legend(["numerical", "Edgeworth", "CLT"], prop={"size": 25})
    pickle_out = open("mixture_primal_n{}_pquarter.pickle".format(num_composition), "wb")
    pickle.dump(
        {
            "f_numerical": f_numerical,
            "f_edgeworth": f_edgeworth,
            "f_eps_delta": f_eps_delta,
            "f_clt": f_clt,
            "epsilon": epsilon,
            "deltas": deltas,
            "clt_time": clt_time,
            "edgeworth_time": edgeworth_time,
            "numerical_time": numerical_time,
        },
        pickle_out,
    )
    pickle_out.close()


def mixture_primal(num_composition, save_fig=False, p=0.5):
    fig, axs = plt.subplots(1, 1, figsize=(10, 10))
    mu = 1.0  # corresponds to sigma in noisy SGD is 1.0

    def dens_func_P(x):
        return scipy.stats.norm.pdf(x)

    def dens_func_Q(x):
        return p * scipy.stats.norm.pdf(x, loc=mu) + (1 - p) * scipy.stats.norm.pdf(x)

    def log_likelihood_ratio_func(x):
        """
        log_likelihood_ratio(x) = log(dens_func_Q(x) / dens_func_P(x))
        """
        z = mu * x - 0.5 * mu ** 2
        if z > 0:
            return z + np.log(p) + np.log1p((1 / p - 1) * np.exp(-z))
        else:
            return np.log(1 + p * (np.exp(z) - 1))

    run_expr_mixture(
        dens_func_P,
        dens_func_Q,
        log_likelihood_ratio_func,
        num_composition,
        mu_gaussian=mu,
        p=p,
        ax=axs,
        left=-np.inf,
        right=np.inf,
        use_cornish_fisher=False,
        title="n = {}".format(num_composition),
    )

    plt.show()
    if save_fig:
        fig.savefig(
            "../tex/image/mixture_n{}_new_new.pdf".format(num_composition),
            bbox_inches="tight",
        )
