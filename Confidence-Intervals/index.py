from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

def get_bernoulli_dist(p, nsamples):
    b = stats.bernoulli(p)
    xs = b.rvs(nsamples * tests).reshape(nsamples, -1)
    return xs

def get_phat(xs):
    phat = np.mean(xs, axis=0)
    return phat

def get_hoeffding(conf_int, nsamples, phat, p):
    epsilon_n = np.sqrt(np.log(2/(1 - conf_int))* (1/(2*nsamples)))
    pct = np.logical_and(phat - epsilon_n<p, p <= (epsilon_n + phat)).mean() * 100
    pct_lower = [i - epsilon_n for i in phat]
    pct_upper = [i + epsilon_n for i in phat]
    return epsilon_n, pct, pct_lower, pct_upper

def main(p, nsamples, tests, conf_int):
    xs = get_bernoulli_dist(p, nsamples)
    phat = get_phat(xs)
    epsilon_n, pct, pct_lower, pct_upper = get_hoeffding(conf_int, nsamples, phat, p)
    fig, ax = initialize_fig(pct)
    plot_results(fig, ax, nsamples, phat, pct_lower, pct_upper)

def initialize_fig(pct):
    plt.style.use("grayscale")
    fig, ax = plt.subplots()
    fig.suptitle("Confidence Interval")
    ax.set_title(f"Interval:trapped correct values {pct}\\% of the time.")
    ax.set_xlabel("Trial index, [-]")
    ax.set_ylabel("Value of estimate, [-]")
    ax.grid(True)
    ax.set_ylim(0.2, 0.9)
    return fig, ax

def plot_results(fig, ax, nsamples, phat, pct_lower, pct_upper):
    x_range = [i for i in np.arange(0, nsamples)]
    ax.scatter(x_range, phat, c = "k", label = "Point estimates")
    ax.plot(x_range, pct_lower, color = "b", label = "Hoeff.low.")
    ax.plot(x_range, pct_upper, color = "r", label = "Hoeff.upp.")
    ax.fill_between(x_range, pct_lower, pct_upper, color = "g")
    ax.legend(loc = "best")
    plt.show()

if __name__ == "__main__":
    p = 0.5
    nsamples = 100
    tests = nsamples
    conf_int = 0.95
    main(p, nsamples, tests, conf_int)