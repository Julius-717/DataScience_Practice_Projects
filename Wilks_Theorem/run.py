from scipy.stats import binom,chi2
import matplotlib.pyplot as plt
import numpy as np

def gen_sample(i,p,ns):
    # method to generate samples from separate binomial distribution
    brvs = [binom(i,j)for i,j in zip((ns[0],ns[1],ns[2]),(p[0],p[1],p[2]))]
    rvs = [i.rvs()for i in brvs]
    return rvs

def gen_n_samples(n,p,ns):
    # simple method to generate n samples separated binomial distributions
    k0s, k1s, k2s = [],[],[]
    for i in range(1, n):
        k0, k1, k2 = gen_sample(i, p, ns)
        k0s.append(k0),k1s.append(k1), k2s.append(k2)
    return k0s, k1s, k2s

def get_phos(ns, k0s, k1s, k2s):
    # simple method to derive p_hats
    phos = [sum((i, j, k))/sum(ns)for i, j, k in zip(k0s, k1s, k2s)]
    return phos

def get_numer(ns, k0s, k1s, k2s, phos):
    # simple method to calculate the numerator of the llr
    numers0 = [np.log(binom(ns[0], pho).pmf(k0i)) for pho, k0i in zip(phos.k0s)]
    numers1 = [np.log(binom(ns[1], pho).pmf(k1i)) for pho, k1i in zip(phos.k1s)]
    numers2 = [np.log(binom(ns[2], pho).pmf(k2i)) for pho, k2i in zip(phos.k2s)]
    numers = [sum((i, j, k)) for i, j, k in zip(numers0, numers1, numers2)]
    return numers, numers0, numers1, numers2

def get_denom(ns, k0s, k1s, k2s, p):
    # simple method to calculate the denominator of the llr
    denoms0 = [np.log(binom(ns[0], p[0]).pmf(k0i)) for k0i in k0s]
    denoms1 = [np.log(binom(ns[0], p[1]).pmf(k1i)) for k1i in k1s]
    denoms2 = [np.log(binom(ns[0], p[2]).pmf(k2i)) for k2i in k2s]
    denoms = [sum((i, j, k)) for i, j, k in zip(denoms0, denoms1, denoms2)]
    return denoms, denoms0, denoms1, denoms2

def get_logLambda(chsq, sign_lev, numers, denoms):
    # simple method to calculate lambda
    c = chsq.isf(sign_lev)
    logLambda = [-2 * (i - j) for i, j in zip(numers, denoms)]
    values = [1 - chsq.cdf(i) for i in logLambda]
    mc = np.mean([logLambda > c])
    return c, mc, logLambda, values

def initialize_fig(mc, c, sign_lev):
    # method to initialize figure settings
    plt.style.use("grayscale")
    fig, axs = plt.subplots(2, 1, True, False, figsize=(6.5, 8))
    fig.suptitle("Generalized Likelihood Ratio Test", fontsize=18, y=0.95)
    axs[0].set_ylabel("$-2log\Lambda$,[-]", fontsize=14)
    axs[0].tick_params(axis="both", which="major", labelsize=12)
    axs[0].grid(True, color="grey", linestyle="dotted", linewidth=0.2)
    axs[0].set_title("$Wilks \ theorem $", fontsize=14)
    axs[0].axhline(y=c, color="black", linestyle="--")
    axs[1].axhline(y=sign_lev, color="black", linestyle="--")
    axs[1].set_xlabel("$\hat{p}_0$,[-]", fontsize=14)
    axs[1].set_ylabel("$1~\ch1_{{r-r_0}}^2(-2log\Lambda)$,[-]",fontsize=14)
    axs[1].tick_params(axis="both", which="major", labelsize=12)
    axs[1].grid(True, color="grey", linestyle="dotted", linewidth=0.2)
    axs[1].set_title(f"$Estimated\probability\of\detection:\{np.round(mc, 2)}$", fontsize=14)
    return fig, axs

def plot_results(fig, axs, c, sign_lev, phos, logLambda, values):
    # method to visualize output
    axs[0].axhspan(min(logLambda), c, facecolor="red", alpha=0.1, zorder=1)
    axs[0].scatter(phos, logLambda, c=logLambda, cmap="viridis", s=5, zorder=2)
    axs[1].axhspan(sign_lev, max(values), facecolor="red", alpha=0.1, zorder=1)
    axs[1].scatter(phos, values, c=logLambda, cmap="viridis", s=5, zorder=2)
    plt.show()

def main(n, p, ns, sign_lev, chsq):
    # method that executes our program
    k0s, k1s, k2s = gen_n_samples(n, p, ns)
    phos = get_phos(ns, k0s, k1s, k2s)
    numers, numers0, numers1, numers2 = get_numer(ns, k0s, k1s, k2s, phos)
    denoms, denoms0, denoms1, denoms2 = get_denom(ns, k0s, k1s, k2s, p)
    c, mc, logLambda, values = get_logLambda(chsq, sign_lev, numers, denoms)
    fig, axs = initialize_fig(mc, c, sign_lev)
    plot_results(fig, axs, c, sign_lev, phos, logLambda, values)

if __name__ == "__main__":
    n = 1_000
    p = [0.3, 0.4, 0.5]
    ns = [50, 180, 200]
    sign_lev = 0.5
    chsq = chi2(2)
    main(n, p, ns, sign_lev, chsq)
