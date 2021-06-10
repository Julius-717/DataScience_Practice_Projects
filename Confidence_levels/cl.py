import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

def get_bernoulli_rv():
    coins = [0, 1]
    k = random.choice(coins)
    return k

def get_sym_sum(distance, n, p):
    g = binom.pmf(np.arange(-distance, +distance) + (n/2), n, p).sum()
    return g

def get_verification(tries, n, p, distance):
    out = []
    for i in range(tries):
        ks = [get_bernoulli_rv() for i in range(1, n+1)]
        phat = np.mean(ks)
        out.append(abs(phat-p) < p * 2 * (distance/100))
    return out

def initialize_fig():
    plt.style.use("grayscale")
    fig, ax = plt.subplots()
    fig.suptitle("Binomial Distribution-PMF")
    ax.set_xlabel("$k$, [-]")
    ax.set_ylabel("$f(k, n, p)$, [-]")
    ax.tick_params(axis = "both", which = "major")
    ax.grid(True)
    return fig, ax

def main(n, p, distance, tries):
    x = range(0, n)
    binom_pmf = binom.pmf(x, n, p)
    g = round(get_sym_sum(distance, n, p), 3)
    out = np.mean(get_verification(tries, n, p, distance))
    fig, ax = initialize_fig()
    ax.scatter(x, binom_pmf, c ="red", label = f"p={p}, n={n}")
    ax.vlines(x, [0], binom_pmf, colors = "black")
    ax.vlines(-distance + (n/2), [0], max(binom_pmf), colors = "green", label = "Conf.interval:$\epsilon=0.20$")
    ax.vlines(distance + (n/2), [0], max(binom_pmf), colors = "green")
    ax.set_title(f"$P(|\hat p-p|\feq\epsilon p)\Rightarrow Binomial\sim{g}\|\Bernoulli\sim{out}$")
    ax.set_xlim((3*(n/10)), n-(3*(n/10)))
    ax.legend(loc = "best")
    plt.show()

if __name__ == '__main__':
    n, p, distance, tries = 100, 0.5, 10, 500
    main(n, p, distance, tries)