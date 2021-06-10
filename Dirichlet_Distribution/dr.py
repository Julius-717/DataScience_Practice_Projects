import numpy as np
import scipy.special as sc
import matplotlib.pyplot as plt
from scipy.stats import dirichlet
from mpl_toolkits.mplot3d import Axes3D

def initialize_fig(n, alphas):
    # a simple method of initializing figure
    plt.style.use("grayscale")
    fig = plt.figure(figsize=(8, 6))
    fig.suptitle("Dirichlet distribution")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("$\mu_1$,[-]")
    ax.set_ylabel("$\mu_1$,[-]")
    ax.set_zlabel("$\mu_1$,[-]")
    #set azimuth and elevation
    ax.view_init(30, -330)
    return fig, ax

def make_dir_pdf(n, alphas):
    d = dirichlet(alphas)

    rv = d.rvs(n)

    yy = []
    
    for i in range(n):
        beta_num = np.multiply.reduce([sc.gamma(alpha) for alpha in alphas])
        beta_denom = sc.gamma(np.sum(alphas))
        beta = beta_num / beta_denom
        eq = [(x**(alpha - 1))for alpha, x in zip(alphas, rv[i])]
        dirichlet_pdf = (1/beta) * np.multiply.reduce(eq)
        yy.append(dirichlet_pdf)
    return rv, yy

if __name__ == '__main__':
    #number of generated samples
    n = 5000

    # alphas(X, y, z)
    alphas = [1, 1, 1]

    # call function that calculates dirichlet PDF
    rv, yy = make_dir_pdf(n, alphas)

    # initialize figure

    fig, ax = initialize_fig(n, alphas)

    xs = [rv[index][0] for index, value in enumerate(yy)]
    ys = [rv[index][1] for index, value in enumerate(yy)]
    zs = [rv[index][2] for index, value in enumerate(yy)]
    im =ax.scatter(xs, ys, zs, c=yy, cmap="viridis", alpha=0.8, s=2)

    fig.colorbar(im, ax=ax, label="PDF,[-]")
    plt.show()