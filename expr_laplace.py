import numpy as np
import matplotlib.pyplot as plt
import edgeworth
import scipy.stats
import time
import pickle

from matplotlib import rc

rc("text", usetex=True)


def run_expr(
    dens_func_P,
    dens_func_Q,
    log_likelihood_ratio_func,
    num_composition,
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
    f_clt = edgeworth.compute_gdp_clt(alpha, mu_f)
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
    print("\t\tEdgeworth used {} seconds.".format(edgeworth_time))
    f_edgeworth = np.array(f_edgeworth)

    epsilon = np.linspace(-6, 6, 100)
    start = time.time()
    f_numerical, f_eps_delta, deltas = edgeworth.approx_f_numerical(
        dens_func_Q,
        log_likelihood_ratio_func,
        num_composition,
        epsilon,
        alpha=alpha,
        left=left,
        right=right,
    )
    numerical_time = time.time() - start
    print("\t\tNumerical used {} seconds.".format(numerical_time))

    if ax is not None:
        line1, = ax.plot(alpha, f_numerical, linewidth=4, color="k", linestyle="-")
        line2, = ax.plot(alpha, f_edgeworth, linewidth=4, color="r")
        line3, = ax.plot(alpha, f_clt, linewidth=4, color="b", linestyle="--")
        ax.set_xlabel(r"Type I Error", fontsize=30)
        ax.set_ylabel(r"Type II Error", fontsize=30)
        ax.xaxis.set_tick_params(size=18, labelsize=30)
        ax.yaxis.set_tick_params(size=18, labelsize=30)
        if title is not None:
            ax.set_title(title, fontdict={"fontsize": 30})
        if log_scale:
            ax.set_yscale("log")
        if zoom_in:
            from mpl_toolkits.axes_grid1.inset_locator import (
                zoomed_inset_axes,
                mark_inset,
            )

            axins = zoomed_inset_axes(
                ax, 1.8, loc=1
            )  # zoom-factor: 2.5, location: upper-left
            axins.plot(alpha, f_numerical, linewidth=4, color="k", linestyle="-")
            axins.plot(alpha, f_edgeworth, linewidth=4, color="r")
            axins.plot(alpha, f_clt, linewidth=4, color="b", linestyle="--")
            axins.set_xticks([])
            axins.set_yticks([])
            x1, x2, y1, y2 = 0.0, 0.35, 0, 0.4  # specify the limits
            axins.set_xlim(x1, x2)  # apply the x-limits
            axins.set_ylim(y1, y2)  # apply the y-limits
            mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
        else:
            if num_composition == 1:
                ax.legend(["numerical", "Edgeworth", "CLT"], prop={"size": 30})

    pickle_out = open("laplace_primal_n{}.pickle".format(num_composition), "wb")
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

    return clt_time, edgeworth_time, numerical_time, alpha, f_edgeworth


def run_expr_dual(
    dens_func_P,
    dens_func_Q,
    log_likelihood_ratio_func,
    num_composition,
    vertical_line_x=None,
    use_cornish_fisher=False,
    left=-np.inf,
    right=np.inf,
    title=None,
    ax=None,
    log_scale=False,
):

    moments_P, errs = edgeworth.compute_moments(
        log_likelihood_ratio_func, dens_func_P, max_order=4, left=left, right=right
    )
    kappas_P = edgeworth.compute_cumulants(moments_P)

    # odd order moment will flip the sign under Q
    moments_Q, errs = edgeworth.compute_moments(
        log_likelihood_ratio_func, dens_func_Q, max_order=4, left=left, right=right
    )
    kappas_Q = edgeworth.compute_cumulants(moments_Q)
    print("Cumulants of Q: {}".format(kappas_Q))

    sigma_square_P = [kappas_P[1]] * num_composition
    sigma_square_Q = [kappas_Q[1]] * num_composition
    kappa_3_P = [kappas_P[2]] * num_composition
    kappa_3_Q = [kappas_Q[2]] * num_composition
    kappa_4_P = [kappas_P[3]] * num_composition
    kappa_4_Q = [kappas_Q[3]] * num_composition

    mu_f = (
        (moments_Q[0] - moments_P[0]) / np.sqrt(kappas_P[1]) * np.sqrt(num_composition)
    )
    alpha = np.sort(
        np.concatenate(
            (
                np.logspace(-8, -1, 100),
                np.linspace(0.09, 0.99, 500),
                1.0 - np.logspace(-8, -1, 200),
            ),
            axis=None,
        )
    )

    f_edgeworth = np.zeros(len(alpha))
    start = time.time()
    for ii in range(len(alpha)):
        aa = alpha[ii]
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
        f_edgeworth[ii] = val
    edgeworth_time = time.time() - start
    print("Edgeworth used {} seconds.".format(edgeworth_time))

    epsilon = np.linspace(-6, 10, 100)
    start = time.time()
    f_numerical, f_eps_delta, deltas = edgeworth.approx_f_numerical(
        dens_func_Q, log_likelihood_ratio_func, num_composition, epsilon, alpha=alpha
    )
    numerical_time = time.time() - start
    print("Numerical used {} seconds.".format(numerical_time))

    delta_clt = []
    delta_edgeworth = []
    for eps in epsilon:
        delta_clt.append(
            scipy.stats.norm.cdf(-eps / mu_f + mu_f / 2)
            - np.exp(eps) * scipy.stats.norm.cdf(-eps / mu_f - mu_f / 2)
        )
        delta_edgeworth.append(
            1 + edgeworth.compute_f_conjugate(-np.exp(eps), alpha, f_edgeworth)
        )

    if ax is not None:
        line1, = ax.plot(epsilon, deltas, linewidth=4, color="k", linestyle="-")
        line2, = ax.plot(epsilon, delta_edgeworth, linewidth=4, color="r")
        line3, = ax.plot(epsilon, delta_clt, linewidth=4, color="b", linestyle="--")
        if vertical_line_x is not None:
            ax.axvline(
                x=vertical_line_x,
                ymin=-0.01,
                ymax=0.95,
                linewidth=4,
                linestyle="-.",
                color="darkgray",
            )
        ax.set_xlabel(r"$\epsilon$", fontsize=30)
        ax.set_ylabel(r"$\delta(\epsilon)$", fontsize=30)
        if num_composition == 1:
            ax.legend(
                ["numerical", "Edgeworth", "CLT", r"$x=n\theta$"],
                prop={"size": 30},
                loc="lower left",
            )
        ax.xaxis.set_tick_params(size=18, labelsize=30)
        ax.yaxis.set_tick_params(size=18, labelsize=30)
        if log_scale:
            ax.set_yscale("log")
        if title is not None:
            ax.set_title(title, fontdict={"fontsize": 30})
    pickle_out = open("laplace_dual_n{}.pickle".format(num_composition), "wb")
    pickle.dump(
        {
            "f_numerical": f_numerical,
            "f_edgeworth": f_edgeworth,
            "f_eps_delta": f_eps_delta,
            "epsilon": epsilon,
            "deltas": deltas,
            "delta_clt": delta_clt,
            "delta_edgeworth": delta_edgeworth,
            "edgeworth_time": edgeworth_time,
            "numerical_time": numerical_time,
        },
        pickle_out,
    )
    pickle_out.close()
    # return alpha, f_edgeworth, f_clt, f_numerical


def compare_cornish_fisher(
    dens_func_P,
    dens_func_Q,
    log_likelihood_ratio_func,
    num_composition,
    left=-np.inf,
    right=np.inf,
    title=None,
    ax=None,
):
    moments_P, errs = edgeworth.compute_moments(
        log_likelihood_ratio_func, dens_func_P, max_order=4, left=left, right=right
    )
    kappas_P = edgeworth.compute_cumulants(moments_P)

    # odd order moment will flip the sign under Q
    moments_Q, errs = edgeworth.compute_moments(
        log_likelihood_ratio_func, dens_func_Q, max_order=4, left=left, right=right
    )
    kappas_Q = edgeworth.compute_cumulants(moments_Q)
    print("Cumulants of Q: {}".format(kappas_Q))

    sigma_square_P = [kappas_P[1]] * num_composition
    sigma_square_Q = [kappas_Q[1]] * num_composition
    kappa_3_P = [kappas_P[2]] * num_composition
    kappa_3_Q = [kappas_Q[2]] * num_composition
    kappa_4_P = [kappas_P[3]] * num_composition
    kappa_4_Q = [kappas_Q[3]] * num_composition

    mu_f = 2 * moments_P[0] / np.sqrt(kappas_P[1]) * np.sqrt(num_composition)
    alpha = np.linspace(1e-7, 1 - 1e-7, 100)

    f_edgeworth = []
    f_cornish_fisher = []
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
            use_cornish_fisher=False,
        )
        f_edgeworth.append(val)

        val1 = edgeworth.approx_f_edgeworth(
            aa,
            sigma_square_P,
            sigma_square_Q,
            kappa_3_P,
            kappa_3_Q,
            kappa_4_P,
            kappa_4_Q,
            mu_f,
            use_cornish_fisher=True,
        )
        f_cornish_fisher.append(val1)

    if ax is not None:
        line1, = ax.plot(alpha, f_edgeworth, color="r", linewidth=2)
        line2, = ax.plot(
            alpha, f_cornish_fisher, color="navy", linestyle="-.", linewidth=2
        )
        ax.legend(["Edgeworth Inverse", "Cornish_Fisher"], prop={"size": 20})
        ax.xaxis.set_tick_params(size=10, labelsize=10)
        ax.yaxis.set_tick_params(size=10, labelsize=10)
        if title is not None:
            ax.set_title(title, fontdict={"fontsize": 25})


def laplace_mu3_primal_all(num_compositions, save_fig=False):

    fig, axs = plt.subplots(2, 2, figsize=(20, 20))

    for ii in range(len(num_compositions)):

        num_composition = num_compositions[ii]

        mu = 3.0 / np.sqrt(num_composition)

        def dens_func_p(x):
            return 0.5 * np.exp(-np.abs(x))

        def dens_func_q(x):
            return 0.5 * np.exp(-np.abs(x + mu))

        def log_likelihood_ratio_func(x):
            """
            log_likelihood_ratio(x) = log(dens_func_q(x) / dens_func_p(x))
            """
            return np.abs(x) - np.abs(x + mu)

        run_expr(
            dens_func_p,
            dens_func_q,
            log_likelihood_ratio_func,
            num_composition,
            title="n = {}".format(num_composition),
            ax=axs[int(ii / 2)][ii % 2],
            log_scale=False,
            zoom_in=True if num_composition == 5 else False,
        )

    plt.show()
    if save_fig:
        fig.savefig("../tex/image/laplace_mu3_all.pdf", bbox_inches="tight")


def laplace_mu3_primal(num_composition, save_fig=False):
    fig, axs = plt.subplots(1, 1, figsize=(10, 10))
    mu = 3.0 / np.sqrt(num_composition)

    def dens_func_P(x):
        return 0.5 * np.exp(-np.abs(x))

    def dens_func_Q(x):
        return 0.5 * np.exp(-np.abs(x + mu))

    def log_likelihood_ratio_func(x):
        """
        log_likelihood_ratio(x) = log(dens_func_Q(x) / dens_func_P(x))
        """
        return np.abs(x) - np.abs(x + mu)

    _, _, _, alpha, f_edgeworth = run_expr(
        dens_func_P, dens_func_Q, log_likelihood_ratio_func, num_composition, ax=axs
    )

    plt.show()
    if save_fig:
        fig.savefig(
            "../tex/image/laplace_mu3_n{}.pdf".format(num_composition),
            bbox_inches="tight",
        )
    return alpha, f_edgeworth


def laplace_mu3_dual(num_composition, save_fig=False, log_scale=False):
    fig, axs = plt.subplots(1, 1, figsize=(10, 10))
    mu = 3.0 / np.sqrt(num_composition)

    def dens_func_P(x):
        return 0.5 * np.exp(-np.abs(x))

    def dens_func_Q(x):
        return 0.5 * np.exp(-np.abs(x + mu))

    def log_likelihood_ratio_func(x):
        """
        log_likelihood_ratio(x) = log(dens_func_Q(x) / dens_func_P(x))
        """
        return np.abs(x) - np.abs(x + mu)

    run_expr_dual(
        dens_func_P,
        dens_func_Q,
        log_likelihood_ratio_func,
        num_composition,
        ax=axs,
        log_scale=log_scale,
        vertical_line_x=num_composition * mu,
    )

    plt.subplots_adjust(left=0.15)
    plt.show()
    if save_fig:
        fig.savefig(
            "../tex/image/laplace_mu3_n{}_dual.pdf".format(num_composition),
            bbox_inches="tight",
        )


def laplace_mu3_dual_all(num_compositions, save_fig=False, log_scale=False):

    fig, axs = plt.subplots(2, 2, figsize=(20, 20))

    for ii in range(len(num_compositions)):

        num_composition = num_compositions[ii]

        mu = 3.0 / np.sqrt(num_composition)

        def dens_func_P(x):
            return 0.5 * np.exp(-np.abs(x))

        def dens_func_Q(x):
            return 0.5 * np.exp(-np.abs(x + mu))

        def log_likelihood_ratio_func(x):
            """
            log_likelihood_ratio(x) = log(dens_func_Q(x) / dens_func_P(x))
            """
            return np.abs(x) - np.abs(x + mu)

        run_expr_dual(
            dens_func_P,
            dens_func_Q,
            log_likelihood_ratio_func,
            num_composition,
            title="n = {}".format(num_composition),
            ax=axs[int(ii / 2)][ii % 2],
            log_scale=log_scale,
            vertical_line_x=num_composition * mu,
        )
    plt.show()
    if save_fig:
        fig.savefig("../tex/image/laplace_mu3_dual_all.pdf", bbox_inches="tight")


def laplace_cornish_fisher_vs_numerical_inverse():
    num_compositions = [1, 2, 3, 4]

    fig, axs = plt.subplots(1, 4, figsize=(40, 10))

    for ii in range(len(num_compositions)):
        num_composition = num_compositions[ii]
        mu = 3.0 / np.sqrt(num_composition)
        ax = axs[ii]

        def dens_func_P(x):
            return 0.5 * np.exp(-np.abs(x))

        def dens_func_Q(x):
            return 0.5 * np.exp(-np.abs(x + mu))

        def log_likelihood_ratio_func(x):
            """
            log_likelihood_ratio(x) = log(dens_func_Q(x) / dens_func_P(x))
            """
            return np.abs(x) - np.abs(x + mu)

        compare_cornish_fisher(
            dens_func_P,
            dens_func_Q,
            log_likelihood_ratio_func,
            num_composition,
            title="n = {}".format(num_composition),
            ax=ax,
        )
        ax.tick_params(axis="x", labelsize=20, labelrotation=0)
        ax.tick_params(axis="y", labelsize=20, labelrotation=0)
    plt.show()
    fig.savefig("../tex/image/cornish_fisher.pdf", bbox_inches="tight")
