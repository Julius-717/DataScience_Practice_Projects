import random
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def get_mle(theta, n, tries):
    # function to estimate mle without calculus
    rv = stats.uniform(0, theta)
    mle = rv.rvs((n, tries)).max(0)
    return mle

def initialize_fig(mle):
    # method to initialize figure settings
    plt.style.use("grayscale")
    fig, ax = plt.subplots()
    mean = round(np.mean(mle), 2)
    var = round(np.var(mle), 2)
    fig.suptitle(f"MLE no calc. \n $E(\\theta_{{ML}}$:{mean},$V(\\theta_{{ML}})$:{var}")
    ax.set_xlabel("$\\theta_{ML}$,[-]")
    ax.set_ylabel("$PDF, and\MLES, [-]")
    ax.tick_params(axis="both", which="major")
    ax.grid(True)
    return fig, ax


def main(theta, n, tries):
    # program that executes our program
    mle = get_mle(theta, n, tries)
    theta_ml = mle
    pdf_ml = [j*(i**(j-1))*(theta**(-j)) for i, j in zip(theta_ml, range(tries))]
    fig, ax = initialize_fig(mle)
    ax.hist(mle, label = "MLE", color = "grey")
    ax.scatter(theta_ml, pdf_ml, c = "red", label = "PDF")
    ax.legend(loc = "best")
    plt.show()

if __name__ == '__main__':
    theta = 2
    n = 100
    tries = 500
    main(theta, n, tries)
